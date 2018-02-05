'''
PyTorch implementation of Capsule Networks
'''

import argparse
import os
import shutil
import time

import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

import models.capsNet as capsNet
from dataset.cityscapesDataLoader import cityscapesDataset
import utils

parser = argparse.ArgumentParser(description='PyTorch CapsNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', default=64, type=int,
            help='mini-batch size (default: 64)')
parser.add_argument('--imageSize', default=128, type=int,
            help='height/width of the input image to the network')
parser.add_argument('--lr', default=0.001, type=float,
            help='learning rate (default: 0.0005)')
parser.add_argument('--net', default='',
            help="path to net (to continue training)")
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
            help='print frequency (default:1)')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the trained models',
            default='save_temp', type=str)
parser.add_argument('--verbose', default = False, type=bool,
            help='Prints certain messages which user can specify if true')
parser.add_argument('--routing_iterations', type=int, default=3,
            help='Number of Routing Iterations')
parser.add_argument('--with_reconstruction', action='store_true', default=True,
            help='Net with reconstruction or not')

use_gpu = torch.cuda.is_available()

def main():
    global args
    args = parser.parse_args()
    print(args)

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cudnn.benchmark = True

    # Initialize the data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Scale((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Scale((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]),
    }

    # Data Loading
    data_dir = '/media/salman/DATA/General Datasets/cityscapes'
    # json path for class definitions
    json_path = '/home/salman/pytorch/capsNet/dataset/cityscapesClasses.json'

    image_datasets = {x: cityscapesDataset(data_dir, x, data_transforms[x],
                    json_path) for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # Initialize the Network
    model = capsNet.CapsNet(args.routing_iterations)

    if args.with_reconstruction:
        reconstruction_model = capsNet.ReconstructionNet(16, 20)
        reconstruction_alpha = 0.0005
        model = capsNet.CapsNetWithReconstruction(model, reconstruction_model)

    if use_gpu:
        model.cuda()

    print(model)
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get the dictionary for the id and RGB value pairs for the dataset
    classes = image_datasets['train'].classes
    key = utils.disentangleKey(classes)

    # Initialize the loss function
    # loss_fn = capsNet.MarginLoss(0.9, 0.1, 0.5)

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train(dataloaders['train'], model, optimizer, epoch, key)

        # Save checkpoints
        #torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (args.save_dir, epoch))

def train(train_loader, model, optimizer, epoch, key):
    '''
        Run one training epoch
    '''
    model.train()
    for i, (data, target) in enumerate(train_loader):

        # Generate the target vector from the groundtruth image
        # Multiplication by 255 to convert from float to unit8
        target_temp = target * 255
        label = utils.generateGTmask(target_temp, key)
        print(torch.max(label))

        if use_gpu:
            data = data.cuda()
            label = label.cuda()

        #gt.view(-1)
        #print(target)
        data, label = Variable(data), Variable(label, requires_grad=False)
        label = label.float()
        optimizer.zero_grad()
        if args.with_reconstruction:
            output, probs = model(data, label)
            loss = F.mse_loss(output, label)
            # margin_loss = loss_fn(probs, target)
            # loss = reconstruction_alpha * reconstruction_loss + margin_loss

        # if args.verbose:
        print(output[0,3000:3020])
        print(label[0,3000:3020])

        loss.backward()
        optimizer.step()

        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, args.epochs, i, len(train_loader), loss.mean().data[0]))
        if i % args.print_freq == 0:
        #    vutils.save_image(real_cpu,
        #            '%s/real_samples.png' % args.save_dir,
        #            normalize=True)
        #    #fake = netG(fixed_noise)
        #    vutils.save_image(fake.data,
        #            '%s/fake_samples_epoch_%03d.png' % (args.save_dir, epoch),
        #            normalize=True)
            utils.displaySamples(data, output, target, use_gpu, key)

if __name__ == '__main__':
    main()
