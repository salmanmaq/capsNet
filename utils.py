'''
Utilites for data visualization and manipulation.
'''

import torch
import numpy as np
import cv2

def displaySamples(real, generated, genSeg, seg_mask, use_gpu):
    ''' Display the original and the reconstructed image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            real image, output image
    '''

    if use_gpu:
        real = real.cpu()
        generated = generated.cpu()
        seg_mask = seg_mask.cpu()
        genSeg = genSeg.cpu()

    #unNorm = UnNormalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    # seg_mask = seg_mask.numpy()
    # seg_mask = seg_mask[0,:,:,:]
    # if seg_mask.shape[2] == 1:
    #     real_depth = real.shape[1]
    #     real_height = real.shape[2]
    #     real_width = real.shape[3]
    #     seg_mask = np.reshape(seg_mask, (real_depth, real_height, real_width))
    # seg_mask = np.transpose(seg_mask, (1,2,0))
    # seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)

    seg_mask = seg_mask.numpy()
    seg_mask = np.transpose(np.squeeze(seg_mask[0,:,:,:]), (1,2,0))
    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)

    generated = generated.data.numpy()
    generated = np.transpose(np.squeeze(generated[0,:,:,:]), (1,2,0))
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
    #output = unNorm(output)

    genSeg = genSeg.data.numpy()
    genSeg = np.transpose(np.squeeze(genSeg[0,:,:,:]), (1,2,0))
    genSeg = cv2.cvtColor(genSeg, cv2.COLOR_BGR2RGB)

    real = real.numpy()
    real = np.transpose(np.squeeze(real[0,:,:,:]), (1,2,0))
    real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
    #real = unNorm(real)

    stacked = np.concatenate((seg_mask, generated, genSeg, real), axis = 1)

    cv2.namedWindow('Seg | Gen | GenSeg | Real', cv2.WINDOW_NORMAL)
    cv2.imshow('Seg | Gen | GenSeg | Real', stacked)

    # cv2.namedWindow('Real Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)
    #
    # cv2.imshow('Real Image', real)
    # cv2.imshow('Reconstructed Image', output)
    # cv2.imshow('Segmentation Mask', seg_mask)

    cv2.waitKey(1)

def disentangleKey(key):
    '''
        Disentangles the key for class and labels obtained from the
        JSON file
        Returns a python dictionary of the form:
            {Class Id: RGB Color Code as numpy array}
    '''
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0,c1,c2])
        dKey[class_id] = color_array

    return dKey

def generateGTmask(batch, key):
    '''
        Generates the category-wise encoded vector for the segmentation classes
        for a batch of images.
        Returns a tensor of size: [batchSize, imgSize**2, 1]
    '''
    batch = batch.numpy()
    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        cat_mask = np.ones((img.shape[0], img.shape[1]))
        # Multiply by 19 since 19 is considered label for the background class
        cat_mask = cat_mask * 19

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = -1))
            cat_mask[mask] = k

        cat_mask = torch.from_numpy(cat_mask).view(-1,1).unsqueeze(0)

        if 'label' in locals():
            label = torch.cat((label, cat_mask), 0)
        else:
            label = cat_mask
        #print('img copy masked')
        #print(img_copy)

    return label
