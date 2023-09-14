from __future__ import print_function
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
import math
from TranslateDatasets_AC_gate import TranslateDatasets
import yaml
import utils
from models import DenoisingDiffusion, DiffusiveRestoration

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='AC with Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='dataset/ckpts/0/PET_dataset_ddpm.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=None,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=1000,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='output/train_24/result/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    
    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))

    # Dataset and DataLoad
    test_ptb_data_txt = 'dataset/splitTxt2/PET_myocardium_test_' + \
        str(config.sampling.fold) + '.txt'
    test_set = TranslateDatasets(test_ptb_data_txt)
    test_loader = DataLoader(
        dataset=test_set, batch_size=config.training.batch_size, num_workers=config.data.num_workers, shuffle=False)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config,config.sampling.fold)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(test_loader, r=None)


if __name__ == '__main__':
    main()

