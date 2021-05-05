import torch
from vit_pytorch import ViT



from __future__ import print_function
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
import models
import data_loader
import transformations


def train_stage_1(config, model, encoder, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    encoder.train()
    correct, total = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(encoder(data))
        _, predicted = torch.max(pred.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss=loss_fn(pred,target)
        loss.backward()
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            wandb.log({"train batch loss": loss.item(), 'epoch': epoch})
            if config.dry_run:
                break
    accuracy = 100. * correct / total
    return accuracy

def test_stage_1(config, model, encoder, device, test_loader, loss_fn, epoch):
    model.eval()
    correct, total, test_loss = 0, 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            outputs = model(encoder(data))
            loss = loss_fn(outputs, target)   # sum up batch loss
            test_loss += loss.sum().item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % config.log_interval == 0:
                wandb.log({"test batch loss": loss.item(), 'epoch': epoch})
                if config.dry_run:
                    break
    accuracy = 100. * correct / total
    test_loss /= len(test_loader.dataset)
    wandb.log({"test loss": test_loss, 'epoch': epoch})
    print('Accuracy test images: %d %%' % (accuracy))
    return accuracy

def stage_1(config, device, image_datasets, classifier, encoder, ssnet, genusnet, familynet, train_kwargs, test_kwargs):
    train_loader = torch.utils.data.DataLoader(image_datasets['train'],**train_kwargs)
    test_loader = torch.utils.data.DataLoader(image_datasets['val'], **test_kwargs)

    opt_params = list(encoder.parameters())+list(classifier.parameters())+list(ssnet.parameters())+list(genusnet.parameters())+list(familynet.parameters())
    optimizer = torch.optim.Adam(opt_params, lr = config.lr)

    loss_fn=torch.nn.CrossEntropyLoss()
    loss_fn_test=torch.nn.CrossEntropyLoss()
    wandb.watch(classifier)
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=config.gamma)
    test_accuracy, best_acc = 0.0, 0.0
    for epoch in tqdm(range(config.epochs)):
        train_accuracy = train_stage_1(config, classifier, encoder, device, train_loader, loss_fn, optimizer, epoch)
        test_accuracy = test_stage_1(config, classifier, encoder, device, test_loader, loss_fn_test, epoch)
        scheduler.step()
        wandb.log({"test accuracy": test_accuracy, 'epoch': epoch})
        wandb.log({"train accuracy": train_accuracy, 'epoch': epoch})
        if(test_accuracy>best_acc):
            best_acc = test_accuracy
            torch.save(encoder.state_dict(),os.path.join(config.result_path, 'encoder_fada_extra.pth'))
            torch.save(classifier.state_dict(), os.path.join(config.result_path, 'classifier_fada_extra.pth'))
        wandb.log({"epoch": epoch})


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
    config.input_size = input_size
    config.numclasses1 = numclasses1
    config.numclasses2 = numclasses2



    tags = ['baseline', f'stage {config.stage}']
    wandb.init(project='triplet-loss', config=config, entity='kelvincr', tags=tags)

    if config.run_name is not None:
        wandb.run.name = config.run_name

    result_path = os.path.join(args.result_path, wandb.run.name)
    config.result_path = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)


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
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transformations.TileHerb(),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val': transforms.Compose([
        transformations.CropField(),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val_photo': transforms.Compose([
        transformations.CropField(),
        transforms.CenterCrop((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ])
}
    
    data_dir = config.data_dir 
    herbarium = os.path.join(data_dir, 'herbarium')
    photo = os.path.join(data_dir, 'photo')

    classifier = models.ClassifierPro().to(device)
    encoder = tmodels.resnet50(pretrained=True).to(device)
    encoder.fc = nn.Sequential()
    ssnet = models.TaxonNet(64).to(device)
    genusnet = models.TaxonNet(510).to(device)
    familynet = models.TaxonNet(151).to(device)

    discriminator = models.DCDPro().to(device)
    discriminator_genus = models.DCDPro().to(device)
    discriminator_family = models.DCDPro().to(device)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
    
    base_mapping = image_datasets['train'].class_to_idx
    class_name_to_id = image_datasets['train'].class_to_idx
    id_to_class_name = {v: k for k, v in class_name_to_id.items()}
    
    siamese_dataset = data_loader.FADADatasetSS(data_dir,
                                    photo,
                                    'train',
                                    image_datasets['train'].class_to_idx,
                                    class_name_to_id,
                                    config.image_size
                                    )

    if(config.stage == 1):
        stage_1(config, device, image_datasets, classifier, encoder, ssnet, genusnet, familynet, train_kwargs, test_kwargs)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FADA')
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
    parser.add_argument('--image-size', type=int, default=224, metavar='N',
                        help='image size')
    parser.add_argument('--run-name', type=str, metavar='N',
                    help='run name')
    parser.add_argument('--data-dir', type=str, metavar='N',
                    help='data directory name, default /dev/shm/dataset/')

    args = parser.parse_args()
    main(args)