'''
Class for loading the cityscapes dataset
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import json
import cv2

class cityscapesDataset(Dataset):
    '''
        cityscapes Dataset
    '''

    def __init__(self, root_dir, type, transform=None, json_path=None):
        '''
        Args:
            root_dir (string): Directory with all the images
            transform(callable, optional): Optional transform to be applied
                                           on a sample
        '''
        self.transform = transform
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest',
                                    'leftImg8bit', type)
        self.gt_dir = os.path.join(root_dir, 'gtFine_trainvaltest',
                                   'gtFine', type)
        self.image_list = []
        # Recursively get the list paths to images in all sub-directories
        for dir, _, files in os.walk(self.img_dir):
            for f in files:
                self.image_list.append(os.path.join(dir, f))

        if json_path:
            # Read the json file containing classes information
            # This is later used to generate masks from the segmented images
            self.classes = json.load(open(json_path))['classes']

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        gt_name = os.path.join(self.gt_dir, img_name[90:-15] + 'gtFine_color.png')
        image = Image.open(img_name)
        image = image.convert('RGB')
        gt = Image.open(gt_name)
        gt = gt.convert('RGB')
        #gt = cv2.imread(gt_name, cv2.cvtColor)
        #print(gt)

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)

        return image, gt
