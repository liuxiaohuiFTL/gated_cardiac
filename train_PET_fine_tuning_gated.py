from __future__ import print_function
import argparse
import numpy as np
import os
import math
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
import models
from torch.cuda.amp import GradScaler as GradScaler
from models import DenoisingDiffusion
from TranslateDatasets_AC_gate import TranslateDatasets

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='PET_dataset.yml',
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    for fold in range(0, 1):
        # data loading
        print("=> using dataset '{}'".format(config.data.dataset))
        # dataset setting
        train_ptb_data_txt = './dataset/splitTxt2/PET_myocardium_train_' + \
            str(fold) + '.txt'
        valid_ptb_data_txt = './dataset/splitTxt2/PET_myocardium_valid_' + \
            str(fold) + '.txt'
        # Dataset and DataLoad
        train_set = TranslateDatasets(train_ptb_data_txt)
        valid_set = TranslateDatasets(valid_ptb_data_txt)
        train_loader = DataLoader(
            train_set, batch_size=config.training.batch_size, num_workers=config.data.num_workers, shuffle=True)
        valid_loader = DataLoader(
            valid_set, batch_size=config.training.batch_size, num_workers=config.data.num_workers, shuffle=True)

        # create model
        print("=> creating denoising-diffusion model...")
        diffusion = DenoisingDiffusion(args, config, fold)
        diffusion.train(train_loader,valid_loader)


if __name__ == "__main__":
    main()

       
