import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
import numpy as np
import pandas as pd
from dataset import TimeDataset, AttackEvaluateSet
from torch.utils.data import DataLoader
from attack import Attacker, fft_compress
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecast_models import TimesNet, Autoformer, FEDformer

MODEL_MAP = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'FEDformer': FEDformer
}


class Trainer:
    """
    The trainer for the model
    Main functions:
    1. train: train the surrogate forecasting model and the attacker
    2. validate: validate the attacked and natural performance
    3. test: train a new forecasting from scratch on the poisoned data
    """

    def __init__(self, config, atk_vars, target_pattern, train_mean, train_std,atk_vars_std,atk_vars_mean,
                 train_data, test_data,val_data, train_data_stamps, test_data_stamps,val_data_stamps, device, num_for_hist=12, num_for_futr=12,
                 trigger_name='TgrGCN'):
        self.config = config
        self.mean = train_mean
        self.std = train_std
        self.trigger_name=trigger_name

        self.atk_vars_std = atk_vars_std
        self.atk_vars_mean = atk_vars_mean

        self.test_data = test_data
        self.val_data = val_data
        self.net = MODEL_MAP[self.config.surrogate_name](self.config.Surrogate).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.device = device

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.warmup = config.warmup

        self.train_data_stamps = train_data_stamps
        self.test_data_stamps = test_data_stamps
        self.val_data_stamps = val_data_stamps

       
        self.num_for_hist = num_for_hist
        self.num_for_futr = num_for_futr

        train_set = TimeDataset(train_data, train_mean, train_std, device, num_for_hist=self.num_for_hist,
                                num_for_futr=self.num_for_futr, timestamps=train_data_stamps)
        channel_features = fft_compress(train_data, 200)
        self.attacker = Attacker(train_set, channel_features, atk_vars,atk_vars_std,atk_vars_mean, config, target_pattern,train_std,train_mean, 
                                 batch_size=self.batch_size, device=device,
                                 trigger_name=trigger_name)
        self.use_timestamps = config.Dataset.use_timestamps

        



        if self.trigger_name == 'InverseTgr':

    

            train_data_atk_vars = train_data[:, atk_vars.cpu().numpy()]
            train_data_inverse=train_data_atk_vars[::-1]
            if train_data_stamps is not None:
                train_data_stamps_inverse = train_data_stamps[::-1]
            else:
                train_data_stamps_inverse = None


          
            self.train_set_inverse = TimeDataset(train_data_inverse, train_mean, train_std, device,
                                    num_for_hist=self.num_for_hist, num_for_futr=self.num_for_futr,
                                    timestamps=train_data_stamps_inverse)

        self.prepare_data()

    def load_attacker(self, attacker_state):
        self.attacker.load_state_dict(attacker_state)

    def save_attacker(self):
        attacker_state = self.attacker.state_dict()
        return attacker_state

    def prepare_data(self):
        self.train_set = self.attacker.dataset
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)



        self.cln_val_set = TimeDataset(self.val_data, self.mean, self.std, self.device,
                                       num_for_hist=self.num_for_hist,
                                       num_for_futr=self.num_for_futr, timestamps=self.val_data_stamps)
        self.cln_val_loader = DataLoader(self.cln_val_set, batch_size=self.batch_size, shuffle=False)
        self.cln_test_set = TimeDataset(self.test_data, self.mean, self.std, self.device,
                                        num_for_hist=self.num_for_hist,
                                        num_for_futr=self.num_for_futr, timestamps=self.test_data_stamps)
        self.cln_test_loader = DataLoader(self.cln_test_set, batch_size=self.batch_size, shuffle=False)


        self.atk_val_set = AttackEvaluateSet(self.attacker, self.val_data, self.mean, self.std, self.device,
                                              num_for_hist=self.num_for_hist, num_for_futr=self.num_for_futr,
                                              timestamps=self.val_data_stamps)
        self.atk_val_loader = DataLoader(self.atk_val_set, batch_size=self.batch_size, shuffle=False,
                                            collate_fn=self.atk_val_set.collate_fn)
        self.atk_test_set = AttackEvaluateSet(self.attacker, self.test_data, self.mean, self.std, self.device,
                                              num_for_hist=self.num_for_hist, num_for_futr=self.num_for_futr,
                                              timestamps=self.test_data_stamps)
        self.atk_test_loader = DataLoader(self.atk_test_set, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=self.atk_test_set.collate_fn)


    def train(self):
        self.attacker.train()

        
        if self.trigger_name == 'InverseTgr':
            self.attacker.train_inverse_tgr(self.train_set_inverse,use_timestamps=self.use_timestamps,EPOCH=20,BATCH_SIZE=256)

        poison_metrics = []
        for epoch in range(self.num_epochs):
            self.net.train()  # ensure dropout layers are in train mode

            if epoch > self.warmup:
                if not hasattr(self.attacker, 'atk_ts'):
                    self.attacker.select_atk_timestamp(poison_metrics)
                self.attacker.sparse_inject_batch()

            poison_metrics = []

            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            for batch_index, batch_data in enumerate(self.train_loader):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                self.optimizer.zero_grad()

                x_des = torch.zeros_like(labels)
                outputs = self.net(encoder_inputs, x_mark, x_des, None)
                outputs = self.train_set.denormalize(outputs)
                loss_per_sample = F.smooth_l1_loss(outputs, labels, reduction='none')
                loss_per_sample = loss_per_sample.mean(dim=(1, 2))

                poison_metrics.append(torch.stack([loss_per_sample.cpu().detach(), idx.cpu().detach()], dim=1))
                loss = loss_per_sample.mean()
                loss.backward()
                self.optimizer.step()


            if epoch > self.warmup and (self.trigger_name == 'TgrGCN' or self.trigger_name == 'InverseTgr'):
                self.attacker.update_trigger_generator_refactored(self.net, use_timestamps=self.use_timestamps,epochId=epoch)

            if self.trigger_name == 'TgrGCN' and epoch>50:
                self.validate(self.net, epoch, self.warmup)
            elif self.trigger_name == 'InverseTgr' and epoch > 30:
                self.validate(self.net, epoch, self.warmup)
            #self.validate(self.net, epoch, self.warmup)

            

    def validate(self, model, epoch, atk_eval_epoch=0, state='train'):
        """
        Evaluate the model's performance on clean and attacked data.
        
        Args:
            model: The model to be evaluated
            epoch: Current training epoch
            atk_eval_epoch: Epoch after which to start evaluating attacks
            state: 'train' or 'test' to indicate which dataset to use
            
        Returns:
            info: String containing evaluation metrics
            atk_mae: Mean Absolute Error on attacked data
        """
        # Set model and attacker to evaluation mode
        model.eval()
        self.attacker.eval()
        
        # Initialize information strings for clean and attacked data metrics
        cln_info = atk_info = ''
        
        # Disable gradient computation for evaluation efficiency
        with torch.no_grad():
            # Lists to store predictions and targets for clean data
            cln_preds = []
            cln_targets = []
            
            # Lists to store predictions and targets for attacked data
            atk_targets = []
            atk_preds = []  
            atk_clean_targets = []  # Targets for non-attacked positions in attacked data
            atk_clean_preds = []    # Predictions for non-attacked positions in attacked data

            # Select appropriate data loader based on training/testing state
            if state == 'train':
                cln_data_loader = self.cln_val_loader
            else:
                cln_data_loader = self.cln_test_loader

            # Process clean validation/test data
            for batch_index, batch_data in enumerate(cln_data_loader):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    # Create dummy timestamp marks if not using timestamps
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data

                # Reshape inputs and labels to appropriate dimensions and move to device
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                # Create dummy desired output tensor
                x_des = torch.zeros_like(labels)
                # Get model predictions
                outputs = model(encoder_inputs, x_mark, x_des, None)
                # Denormalize outputs to original scale
                outputs = self.cln_test_set.denormalize(outputs)

                # Store predictions and targets
                cln_targets.append(labels.cpu().detach().numpy())
                cln_preds.append(outputs.cpu().detach().numpy())

            # Combine results from all batches
            cln_preds = np.concatenate(cln_preds, axis=0)
            cln_targets = np.concatenate(cln_targets, axis=0)
            
            # Calculate evaluation metrics for clean data
            cln_mae = mean_absolute_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1))
            cln_rmse = np.sqrt(mean_squared_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1)))
            cln_info = f' | clean MAE: {cln_mae:.4f}, clean RMSE: {cln_rmse:.4f}'

            # Initialize attacked data MAE to infinity
            atk_mae = float('inf')

            # Evaluate on attacked data only after specified epoch
            if epoch > atk_eval_epoch:
                # Select appropriate attacked data loader
                if state == 'train':
                    atk_data_loader = self.atk_val_loader
                else:
                    atk_data_loader = self.atk_test_loader

                # Process attacked validation/test data
                for batch_index, batch_data in enumerate(atk_data_loader):
                    if not self.use_timestamps:
                        # Unpack batch data for non-timestamp case
                        clean_encoder_inputs, encoder_inputs, labels, clean_labels, idx, delta_t_list = batch_data
                        x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                    else:
                        # Unpack batch data for timestamp case
                        clean_encoder_inputs, encoder_inputs, labels, clean_labels, x_mark, y_mark, idx, delta_t_list = batch_data

                    # Reshape inputs and labels to appropriate dimensions and move to device
                    encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                    labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)  # Shape: (B, T, C)
                    clean_labels = torch.squeeze(clean_labels).to(self.device).permute(0, 2, 1)
                    x_des = torch.zeros_like(labels)

                    # Get model predictions on attacked inputs
                    outputs = model(encoder_inputs, x_mark, x_des, None)
                    outputs = self.atk_test_set.denormalize(outputs)  # Shape: (B, T, C)

                    '''
                    Evaluation on attacked regions (optimized version)
                    '''
                    attack_labels = []
                    attack_outputs = []
                    # Create mask to identify non-attacked regions (initially all True)
                    mask = torch.ones_like(labels, dtype=torch.bool)

                    # Prepare parameters for attack region processing
                    batch_size = labels.shape[0]
                    n_vars = len(self.attacker.atk_vars)
                    pattern_len = self.attacker.pattern_len
                    time_len = labels.shape[1]

                    # Convert delta_t_list to tensor and ensure proper device
                    delta_t_tensor = torch.tensor(delta_t_list, device=self.device, dtype=torch.long)

                    # Calculate start and end indices for all attack regions (vectorized operation)
                    start_idx = delta_t_tensor  # Shape: (B, n_vars)
                    end_idx = start_idx + pattern_len  # Shape: (B, n_vars)

                    # Handle boundary cases where end index exceeds time length
                    end_idx = torch.clamp(end_idx, max=time_len)
                    # Create mask for valid attack regions
                    valid_mask = (start_idx < end_idx)  # Shape: (B, n_vars)

                    # Generate batch indices grid (B, n_vars)
                    batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).repeat(1, n_vars)

                    # Generate variable indices using attacker's configured variables
                    var_indices = self.attacker.atk_vars.unsqueeze(0).repeat(batch_size, 1)

                    # Filter valid indices (flatten to 1D)
                    valid_batch = batch_indices[valid_mask]
                    valid_var = var_indices[valid_mask]
                    valid_start = start_idx[valid_mask]
                    valid_end = end_idx[valid_mask]

                    # Process all valid attack regions
                    for b, v, s, e in zip(valid_batch, valid_var, valid_start, valid_end):
                        # Extract attacked region (maintain batch dimension)
                        attacked_label = labels[b:b + 1, s:e, v].unsqueeze(-1)
                        attacked_output = outputs[b:b + 1, s:e, v].unsqueeze(-1)

                        attack_labels.append(attacked_label)
                        attack_outputs.append(attacked_output)
                        # Mark these positions as attacked in the mask
                        mask[b, s:e, v] = False

                    # Combine results from current batch's attack regions
                    if attack_labels:
                        labels_slice = torch.cat(attack_labels, dim=0)
                        outputs_slice = torch.cat(attack_outputs, dim=0)
                        atk_targets.append(labels_slice.cpu().detach().numpy())
                        atk_preds.append(outputs_slice.cpu().detach().numpy())

                    '''
                    Evaluation on non-attacked positions in attacked data
                    '''
                    # Select only non-attacked positions using the mask
                    clean_labels_selected = torch.masked_select(labels, mask)
                    clean_outputs_selected = torch.masked_select(outputs, mask)
                    
                    if clean_labels_selected.numel() > 0:
                        atk_clean_targets.append(clean_labels_selected.cpu().detach().numpy())
                        atk_clean_preds.append(clean_outputs_selected.cpu().detach().numpy())

                # Calculate evaluation metrics for attacked data
                if atk_targets:
                    atk_targets_np = np.concatenate(atk_targets, axis=0)
                    atk_preds_np = np.concatenate(atk_preds, axis=0)
                    atk_mae = mean_absolute_error(atk_targets_np.reshape(-1, 1), atk_preds_np.reshape(-1, 1))
                    atk_rmse = np.sqrt(mean_squared_error(atk_targets_np.reshape(-1, 1), atk_preds_np.reshape(-1, 1)))
                else:
                    atk_mae = 0.0
                    atk_rmse = 0.0

                # Calculate evaluation metrics for non-attacked positions in attacked data
                if atk_clean_targets:
                    clean_targets_np = np.concatenate(atk_clean_targets, axis=0)
                    clean_preds_np = np.concatenate(atk_clean_preds, axis=0)
                    clean_mae = mean_absolute_error(clean_targets_np.reshape(-1, 1), clean_preds_np.reshape(-1, 1))
                    clean_rmse = np.sqrt(
                        mean_squared_error(clean_targets_np.reshape(-1, 1), clean_preds_np.reshape(-1, 1)))
                else:
                    clean_mae = 0.0
                    clean_rmse = 0.0

                # Compile attack evaluation information
                atk_info = (
                    f' | attacked MAE: {atk_mae:.4f}, attacked RMSE: {atk_rmse:.4f}'
                    f' | clean MAE: {clean_mae:.4f}, clean RMSE: {clean_rmse:.4f}'
                )

            # Combine all evaluation information
            info = f'Epoch: {epoch}' + cln_info + atk_info
            print(info)
            return info, atk_mae




    def test(self, fc_model_path=None):
        ret_str = ''
        best_atk_mae = float('inf')  # Initialize best attack MAE to infinity
        best_ret_str = ''  # Store ret_str corresponding to the best model
        self.attacker.eval()

        # Initialize model and optimizer
        model = MODEL_MAP[self.config.model_name](self.config.Model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Inject attack
        self.attacker.sparse_inject_batch()

        self.train_set = self.attacker.dataset
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training new forecasting model {epoch}/{self.num_epochs}')
            for batch_index, batch_data in enumerate(pbar):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data

                # Adjust data format
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                optimizer.zero_grad()

                # Forward pass
                x_des = torch.zeros_like(labels)
                outputs = model(encoder_inputs, x_mark, x_des, None)
                outputs = self.train_set.denormalize(outputs)

                # Compute loss and backpropagate
                loss = F.smooth_l1_loss(outputs, labels)
                loss.backward()
                optimizer.step()

           
            # # Speed up: validate only after epoch > 50 for TgrGCN
            if epoch > 50 and self.trigger_name == 'TgrGCN':
                current_ret_str, current_atk_mae = self.validate(model, epoch, 0, state='test')

                # Track best model after epoch > 90
                if epoch > 90:
                    if current_atk_mae < best_atk_mae:
                        best_atk_mae = current_atk_mae
                        best_ret_str = current_ret_str
                        if fc_model_path is not None:
                            torch.save(model.state_dict(), fc_model_path)
                            print(f"Best model (epoch {epoch}) saved to {fc_model_path} (atk_mae: {best_atk_mae:.4f})")

            if epoch > 30 and self.trigger_name == 'InverseTgr':
                current_ret_str, current_atk_mae = self.validate(model, epoch, 0, state='test')
                if epoch > 40:
                    if current_atk_mae < best_atk_mae:
                        best_atk_mae = current_atk_mae
                        best_ret_str = current_ret_str
                        if fc_model_path is not None:
                            torch.save(model.state_dict(), fc_model_path)
                            print(f"Best model (epoch {epoch}) saved to {fc_model_path} (atk_mae: {best_atk_mae:.4f})")

        # Return ret_str of best model (or empty string if not reached epoch > 90)
        return best_ret_str if best_ret_str else ret_str
