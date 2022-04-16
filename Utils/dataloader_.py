from __future__ import print_function, division
import os
from secrets import choice
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SpatialMotionLoader(Dataset):
    def __init__(self, 
                 root_dir, 
                 mode = 'train', 
                 transform = None, 
                 img = True, 
                 flo = False,
                 spatial_class = 8,
                 motion_class = 4,
                 spatial_motion_class = 16,
                 siamese = False
                 ):
        assert img or flo, "Either both or one from [flo, img] should be True"
        self.siamese = siamese
        self.s_class = spatial_class
        self.m_class = motion_class
        self.sm_class = spatial_motion_class
        
        self.img = img
        self.flo = flo
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.datalist = range(10240)
    
    def rotateImg(self,img):
        return img
    
    def rotateFlo(self,flo):
        return flo
    
    def processImg(self,img):
        return img
    
    def processFlo(self,flo):
        return flo
    
    def getFlo(self,index):
        flo = torch.rand(2,64*3,64*2).float()
        
        if self.siamese:
            flo_view_1 = self.processFlo(self.rotateFlo(flo))
            flo_view_2 = self.processFlo(self.rotateFlo(flo))
            return flo_view_1, flo_view_2
        
        flo = self.processFlo(flo)
        return flo
    
    def getImg(self,index):
        img = torch.rand(3,64*3,64*2).float()
        
        if self.siamese:
            img_view_1 = self.processImg(self.processImg(img))
            img_view_2 = self.processImg(self.processImg(img))
            return img_view_1, img_view_2
        
        img = self.processImg(img)
        return img
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        sample = {}
        if self.img:
            if self.siamese:
                img_1, img_2 = self.getImg(index)
                sample['img_1'] = img_1
                sample['img_2'] = img_2
            else:    
                sample['img_1'] = self.getImg(index)
            sample['img_label'] = choice(range(self.s_class))
        if self.flo:
            if self.siamese:
                flo_1, flo_2 = self.getFlo(index)
                sample['flo_1'] = flo_1
                sample['flo_2'] = flo_2
            else:
                sample['flo_1'] = self.getFlo(index)
            sample['flo_label'] = choice(range(self.m_class))
        
        if self.img and self.flo:
            sample['label'] = choice(range(self.sm_class))
            
        return sample


def getLoader(root_dir = None,
              mode = 'train',
              batch_size = 1,
              shuffle = True,
              num_workers = 0,
              img = True,
              flo = False,
              siamese = False,
            ):
    return DataLoader(SpatialMotionLoader(root_dir, 
                                          mode = mode, 
                                          img = img, 
                                          flo = flo,
                                          siamese = siamese
                                          ), 
                      batch_size = batch_size,
                      shuffle = shuffle,
                      num_workers = num_workers
                      )

if __name__ == '__main__':
    dt = getLoader(root_dir = None, img = True, flo = True)
    sample = next(iter(dt))
    for s in sample:
        print(s, sample[s].shape)