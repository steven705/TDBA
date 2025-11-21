import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from timefeatures import time_features
import os
import pandas as pd


def load_raw_data(dataset_config):
    """
    Load and preprocess raw dataset, split into training, validation and test sets
    
    Args:
        dataset_config: Configuration object containing dataset parameters
        
    Returns:
        Depending on dataset type, returns mean, std, and split datasets along with timestamps if available
    """
    save_dir = os.path.join('data', 'defense', dataset_config.dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    if 'PEMS' in dataset_config.dataset_name:
        # Load PEMS dataset (traffic data)
        raw_data = np.load(dataset_config.data_filename)['data']
        # Split into train (60%), validation (20%), test (20%)
        train_data_seq = raw_data[:int(0.6 * raw_data.shape[0])]
        val_data_seq = raw_data[int(0.6 * raw_data.shape[0]):int(0.8 * raw_data.shape[0])]
        test_data_seq = raw_data[int(0.8 * raw_data.shape[0]):]

        # Calculate normalization statistics from training data
        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        # Handle scalar case for mean and std
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        # Create labeled DataFrames for visualization/analysis
        columns = [f"feature_{i}" for i in range(raw_data.shape[1])]

        df_test = pd.DataFrame(test_data_seq[:, :, 0], columns=columns)
        df_test['Normal/Attack'] = 'Normal'
        df_test.to_csv(os.path.join(save_dir, 'train_labeled.csv'), index=False)

        df_val = pd.DataFrame(val_data_seq[:, :, 0], columns=columns)
        df_val['Normal/Attack'] = 'Normal'
        df_val.to_csv(os.path.join(save_dir, 'val_labeled.csv'), index=False)

        # Save feature list
        with open(os.path.join(save_dir, 'list.txt'), 'w') as f:
            for col in df_test.columns:
                f.write(col + '\n')

        return train_mean, train_std, train_data_seq, test_data_seq, val_data_seq

    elif dataset_config.dataset_name.startswith('ETT') or dataset_config.dataset_name in ['Weather', 'Electricity', 'Traffic']:
        # Load ETT, Weather, Electricity, or Traffic datasets
        raw_data = pd.read_csv(dataset_config.data_filename)
        raw_data_feats = raw_data.values[:, 1:]  # Extract features (excluding timestamp)
        raw_data_stamps = pd.to_datetime(raw_data.values[:, 0])  # Extract timestamps

        # Split into train (60%), validation (20%), test (20%)
        train_data_seq = raw_data_feats[:int(0.6 * raw_data_feats.shape[0])]
        val_data_seq = raw_data_feats[int(0.6 * raw_data_feats.shape[0]):int(0.8 * raw_data_feats.shape[0])]
        test_data_seq = raw_data_feats[int(0.8 * raw_data_feats.shape[0]):]

        # Split timestamps correspondingly
        train_data_stamps = raw_data_stamps[:int(0.6 * raw_data_stamps.shape[0])]
        val_data_stamps = raw_data_stamps[int(0.6 * raw_data_stamps.shape[0]):int(0.8 * raw_data_stamps.shape[0])]
        test_data_stamps = raw_data_stamps[int(0.8 * raw_data_stamps.shape[0]):]

        # Calculate normalization statistics from training data
        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        # Handle scalar case for mean and std
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        # Create labeled DataFrames for visualization/analysis
        columns = [f"feature_{i}" for i in range(len(raw_data.columns[1:]))]
        df_test = pd.DataFrame(test_data_seq, columns=columns)
        df_test['Normal/Attack'] = 'Normal'
        df_test.to_csv(os.path.join(save_dir, 'train_labeled.csv'), index=False)

        df_val = pd.DataFrame(val_data_seq, columns=columns)
        df_val['Normal/Attack'] = 'Normal'
        df_val.to_csv(os.path.join(save_dir, 'val_labeled.csv'), index=False)

        # Save feature list
        with open(os.path.join(save_dir, 'list.txt'), 'w') as f:
            for col in columns:
                f.write(col + '\n')

        return train_mean, train_std, train_data_seq, test_data_seq, val_data_seq, train_data_stamps, test_data_stamps, val_data_stamps

    else:
        raise ValueError('Dataset not supported')


class TimeDataset(Dataset):
    """
    Dataset class for time series data processing
    
    Handles data normalization, timestamp processing, and data splitting into historical and future segments
    """
    def __init__(self, raw_data, mean, std, device, num_for_hist=12, num_for_futr=12, timestamps=None):
        """
        Initialize time series dataset
        
        Args:
            raw_data: Input time series data with shape (T, n, c) 
                      where T=time steps, n=sensors/variables, c=channels
            mean: Mean value for normalization
            std: Standard deviation for normalization
            device: Computing device (CPU/GPU)
            num_for_hist: Number of time steps for historical context
            num_for_futr: Number of time steps for future prediction
            timestamps: Timestamp information if available
        """
        self.device = device
        self.data = raw_data
        self.use_timestamp = timestamps is not None
        
        # Process timestamps if provided
        if self.use_timestamp:
            self.timestamps = time_features(timestamps)
            self.timestamps = self.timestamps.transpose(1, 0)
            self.timestamps = torch.from_numpy(self.timestamps).float().to(self.device)
        else:
            self.timestamps = None

        # Reshape and transpose data to (n, c, T) format
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1)
        self.data = np.transpose(self.data, (1, 2, 0)).astype(np.float32)
        self.data = torch.from_numpy(self.data).float().to(self.device)

        # Initialize poisoned data as clean data initially
        self.init_poison_data()  

        self.std = float(std)
        self.mean = float(mean)
        self.num_for_hist = num_for_hist  # Historical time steps
        self.num_for_futr = num_for_futr  # Future time steps

        print('Shape of data:', self.data.shape)

    def __len__(self):
        """Return number of samples in dataset"""
        return self.data.shape[-1] - self.num_for_hist - self.num_for_futr + 1

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing input features, poisoned target, clean target, 
            and timestamps if available
        """
        # Get historical data segment
        data = self.poisoned_data[:, 0:1, idx:idx + self.num_for_hist]
        data = self.normalize(data)

        # Get target segments (poisoned and clean)
        poisoned_target = self.poisoned_data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
        clean_target = self.data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
        
        # Return with or without timestamps based on availability
        if not self.use_timestamp:
            return data, poisoned_target, clean_target, idx
        else:
            input_stamps = self.timestamps[idx:idx + self.num_for_hist]
            target_stamps = self.timestamps[idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
            return data, poisoned_target, clean_target, input_stamps, target_stamps, idx
    
    def init_poison_data(self):
        """Initialize poisoned data as a copy of clean data"""
        self.poisoned_data = torch.clone(self.data).detach().to(self.device)

    def normalize(self, data):
        """Normalize data using precomputed mean and std"""
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """Denormalize data using precomputed mean and std"""
        return data * self.std + self.mean


class AttackEvaluateSet(TimeDataset):
    """
    Dataset class for evaluating attack effectiveness
    
    Extends TimeDataset to include attack pattern injection and trigger generation
    """
    def __init__(self, attacker, raw_data, mean, std, device, num_for_hist=12, num_for_futr=12, timestamps=None):
        """
        Initialize attack evaluation dataset
        
        Args:
            attacker: Attack object containing attack parameters and methods
            raw_data: Input time series data
            mean: Mean value for normalization
            std: Standard deviation for normalization
            device: Computing device (CPU/GPU)
            num_for_hist: Number of time steps for historical context
            num_for_futr: Number of time steps for future prediction
            timestamps: Timestamp information if available
        """
        super(AttackEvaluateSet, self).__init__(raw_data, mean, std, device, num_for_hist, num_for_futr, timestamps)
        self.attacker = attacker
        # Initialize delta time lists for consistent attack pattern placement
        self._init_delta_t_lists()

    def _init_delta_t_lists(self):
        """
        Generate fixed delta_t lists for each timestamp to ensure reproducibility
        
        delta_t determines where in the future sequence the attack pattern starts
        """
        # Use fixed seed for reproducibility
        generator = torch.Generator(device='cpu').manual_seed(42)

        n_atk = len(self.attacker.atk_vars)
        pattern_len = self.attacker.pattern_len
        max_delta = self.attacker.fct_output_len - pattern_len

        # Generate delta_t values for each sample
        self.delta_t_lists = []
        for idx in range(len(self)): 
            delta_t_list = []
            for _ in range(n_atk):
                # Randomly generate valid delta_t within range
                delta_t = torch.randint(0, max_delta + 1, (1,), generator=generator).item()
                delta_t_list.append(delta_t)
            self.delta_t_lists.append(delta_t_list)

    def collate_fn(self, data):
        """
        Custom collate function to process batches of data
        
        Handles batching, attack pattern injection, and trigger generation
        
        Args:
            data: List of samples from the dataset
            
        Returns:
            Batch of data with clean and attacked versions
        """
        # Process timestamps if available
        if self.use_timestamp:
            features, target, clean_target, input_stamps, target_stamps, idx = zip(*data)
            input_stamps = torch.stack(input_stamps, dim=0)
            target_stamps = torch.stack(target_stamps, dim=0)
        else:
            features, target, clean_target, idx = zip(*data)
            target_stamps = None

        # Stack tensor lists
        features = torch.stack(features, dim=0)          # [B, C, 1, T_enc]
        clean_target = torch.stack(clean_target, dim=0)  # [B, C, T_dec]

        # Initialize target with clean values before attack injection
        target = clean_target.clone().detach().to(self.device)

        # Denormalize features for attack processing
        features = self.denormalize(features)
        features_seq = features.squeeze(2)  # [B, C, T_enc]

        # Get delta_t lists for current batch
        batch_delta_t_lists = [self.delta_t_lists[i] for i in idx]
        batch_size = features.shape[0]
        n_atk = len(self.attacker.atk_vars)
        pattern_len = self.attacker.pattern_len
        trigger_len = self.attacker.trigger_len

        # Inject attack patterns into target sequence
        for batch_idx in range(batch_size):
            delta_t_list = batch_delta_t_lists[batch_idx]
            for i, var in enumerate(self.attacker.atk_vars):
                dt = delta_t_list[i]
                # Determine baseline value for attack pattern
                if dt == 0:
                    # Use last value from historical sequence
                    base = features_seq[batch_idx, var, -1] 
                else:
                    # Use previous value from target sequence
                    base = target[batch_idx, var, dt-1]
                # Inject attack pattern
                target[batch_idx, var, dt:dt + pattern_len] = self.attacker.target_pattern[i] + base

        # Prepare full sequence for trigger generation
        slice_full = torch.cat([features_seq, target], dim=2)  # [B, C, T_enc + T_dec]
        slice_full = slice_full.permute(1, 0, 2)  # [C, B, T_total]

        # Generate attack triggers using the attacker
        triggers, perturbations = self.attacker.generate_triggers_and_perturbations(
            slice=slice_full,
            delta_t_list=batch_delta_t_lists,
            slice_id=list(idx),
            use_timestamps=self.use_timestamp,
            target_stamps=target_stamps
        )

        # Prepare clean encoder inputs
        clean_encoder_inputs = features.clone().detach().to(self.device)
        clean_encoder_inputs = self.normalize(clean_encoder_inputs)

        # Inject triggers into features
        triggers = triggers.permute(1, 0, 2)  # [B, n_atk, trigger_len]
        triggers = triggers.unsqueeze(2)  # [B, n_atk, 1, trigger_len]
        features[:, self.attacker.atk_vars, :, -trigger_len:] = triggers
        features = self.normalize(features)

        # Return appropriate batch data based on timestamp usage
        if not self.use_timestamp:
            return clean_encoder_inputs, features, target, clean_target, idx, batch_delta_t_lists
        else:
            return clean_encoder_inputs, features, target, clean_target, input_stamps, target_stamps, idx, batch_delta_t_lists
    