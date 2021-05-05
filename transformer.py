import torch
from vit_pytorch import ViT

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models as tmodels
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from tqdm import tqdm
import os
import os.path
import pathlib
from PIL import Image
import datasets

def main(args):
    # Training settings

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    config = wandb.config
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.gamma = args.gamma
    config.epochs = args.epochs
    config.test_batch_size = args.test_batch_size
    config.log_interval = args.log_interval
    config.image_size = args.image_size
    config.dry_run = args.dry_run
    config.num_workers = args.num_workers
    config.stage = args.stage
    config.run_name = args.run_name
    config.data_dir = args.data_dir if args.data_dir is not None else '/dev/shm/dataset'


    # tags = ['baseline', f'stage {config.stage}']
    # wandb.init(project='triplet-loss', config=config, entity='kelvincr', tags=tags)

    # if config.run_name is not None:
    #     wandb.run.name = config.run_name

    # result_path = os.path.join(args.result_path, wandb.run.name)
    # config.result_path = result_path
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)


    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': config.batch_size }
    test_kwargs = {'batch_size': config.test_batch_size }
    if use_cuda:
        cuda_kwargs = {'num_workers': config.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(15),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ])
}
    
    data_dir = config.data_dir 
    herbarium = os.path.join(data_dir, 'herbarium')
    photo = os.path.join(data_dir, 'photo')

    herbarium_dataset = datasets.PlantCLEF(herbarium,transform=data_transforms['train'])
    dloader = torch.utils.data.DataLoader(herbarium_dataset,**train_kwargs)

    v = ViT(
        image_size = config.image_size,
        patch_size = 32,
        num_classes = 997,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    for batch_idx, (img, family, genus, label) in enumerate(dloader):
        preds = v(img)
        print(preds.shape)
        print(label.shape)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Transformer')
    parser.add_argument('--batch-size', type=int, default=60, metavar='N',
                        help='input batch size for training (default: 60)')
    parser.add_argument('--test-batch-size', type=int, default=60, metavar='N',
                        help='input batch size for testing (default: 60)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--result-path', type=pathlib.Path, default="../result/",
                        help='Path for Saving the current Model')
    parser.add_argument('--stage', type=int, default=0, metavar='N',
                        help='stage of model 0 all, 1 classifier, 2 discriminator, 3 final')
    parser.add_argument('--num-workers', type=int, default=6, metavar='N',
                        help='number of workers for data loaders')
    parser.add_argument('--image-size', type=int, default=256, metavar='N',
                        help='image size')
    parser.add_argument('--run-name', type=str, metavar='N',
                    help='run name')
    parser.add_argument('--data-dir', type=str, metavar='N',
                    help='data directory name, default /dev/shm/dataset/')

    args = parser.parse_args()
    main(args)