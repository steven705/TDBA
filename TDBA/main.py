#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import os
import random

from sklearn.preprocessing import StandardScaler

from dataset import load_raw_data
from trainer import Trainer
import yaml
from easydict import EasyDict as edict


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization to ensure experiment reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser_args(model_name, dataset, pattern_type='cone', trigger_name='TgrGCN'):
    # Load configs/default_config.yaml
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)

    # Load training config
    config = yaml.load(open('configs/train_config.yaml'), Loader=yaml.FullLoader)['Train']

    config['model_name'] = model_name
    config['dataset'] = dataset

    # Add experimental parameters
    config['pattern_type'] = pattern_type  # cone, up_trend, up_and_down
    if trigger_name == 'InverseTgr' or trigger_name == 'Random':
        config['attack_lr'] = 0.0001
        config['num_epochs'] = 50

    # Load dataset config
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']

    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']

    config = edict(config)
    return config


def main(config, head_str, trigger_name='TgrGCN', st='train', pattern_type='cone', gpuid=0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0')
    print("CUDA:", USE_CUDA, DEVICE)

    seed_torch()

    data_config = config.Dataset
    print(data_config)
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq, val_data_seq = load_raw_data(data_config)
        train_data_stamps = test_data_stamps = val_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, val_data_seq, train_data_stamps, test_data_stamps, val_data_stamps = load_raw_data(data_config)

    # Standardization
    scaler = StandardScaler()
    # scaler.inverse_transform(train_set_X)
    if not data_config.use_timestamps:
        train_set_X = scaler.fit_transform(train_data_seq[:, :, 0])
    else:
        train_set_X = scaler.fit_transform(train_data_seq)

    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)
    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)

    # Latest version of the target_pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE)
    std = torch.tensor(scaler.var_ ** 0.5).float().to(DEVICE)
    mean = torch.tensor(scaler.mean_).float().to(DEVICE)

    atk_vars_std = std[atk_vars]
    atk_vars_mean = mean[atk_vars]
    print('atk_vars_std', atk_vars_std)
    print('atk_vars_mean', atk_vars_mean)

    print(train_mean, train_std)

    result = torch.zeros(atk_vars.shape[0], target_pattern.shape[0]).to(DEVICE)
    for index, val in enumerate(atk_vars):
        result[index, :] = target_pattern * std[val]
    target_pattern = result

    # Original version
    exp_trainer = Trainer(config, atk_vars, target_pattern, train_mean, train_std, atk_vars_std, atk_vars_mean,
                          train_data_seq, test_data_seq, val_data_seq,
                          train_data_stamps, test_data_stamps, val_data_stamps, DEVICE,
                          num_for_hist=config.Model.seq_len, num_for_futr=config.Model.pred_len,
                          trigger_name=trigger_name)

    # Define save path
    save_dir = f'./checkpoints/exp{head_str}'
    save_file = os.path.join(
        save_dir,
        f'attacker_{trigger_name}_{pattern_type}_{config.dataset}_{config.Model.seq_len}_{config.Model.pred_len}_{config.trigger_len}.pth'
    )

    # Directly use an existing trigger
    if trigger_name == 'Random':
        save_file = os.path.join(
            save_dir,
            f'attacker_TgrGCN_{pattern_type}_{config.dataset}_{config.Model.seq_len}_{config.Model.pred_len}_{config.trigger_len}.pth'
        )

    # Check whether the model file exists
    if os.path.exists(save_file):
        state = torch.load(save_file)
        exp_trainer.load_attacker(state)
        print('load attacker from', save_file)
    else:
        print('=' * 20, ' [ Stage 1 ] ', '=' * 20)
        print('start training surrogate model and attacker')
        exp_trainer.train()

        state = exp_trainer.save_attacker()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(state, save_file)
        print('saved attacker to', save_file)

    # Stage 2
    print('=' * 20, ' [ Stage 2 ] ', '=' * 20)
    print('start evaluating attack performance on a new model')

    fc_model_path = os.path.join('trained_models', "FC_" + head_str, trigger_name, pattern_type,
                                 config.model_name, config.dataset)
    if not os.path.exists(fc_model_path):
        os.makedirs(fc_model_path)
    fc_model_path = os.path.join(fc_model_path, 'fc_model.pth')
    ret_str = exp_trainer.test(fc_model_path=fc_model_path)
    return ret_str


def main_train(model_list, data_list, pattern_type='cone', trigger_name='TgrGCN', day_str='0725',
               chSt='ch', gpuid='0'):

    for data_name in data_list:
        for model_name in model_list:

            config = parser_args(model_name, data_name, pattern_type=pattern_type, trigger_name=trigger_name)
            info = '=' * 20, model_name, '***', data_name, '=' * 20
            print(info)
            head_str = day_str + "_" + chSt
            ret_str = main(config, head_str, trigger_name, pattern_type=pattern_type, gpuid=gpuid)
            with open('result_' + day_str + '_' + pattern_type + '.txt', 'a') as f:
                f.write(str(info))
                f.write('\n')
                f.write(ret_str)


model_list = ['Autoformer']
data_list = ['PEMS03']
pattern_type = 'cone'
trigger_name = 'TgrGCN'
day_str = 'XXXX'
st = 'train'
chSt = 'ch'
gpuid = '7'
main_train(model_list, data_list, pattern_type=pattern_type, trigger_name=trigger_name,
           day_str=day_str, chSt=chSt, gpuid=gpuid)
