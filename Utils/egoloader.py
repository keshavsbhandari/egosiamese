from Utils.egogetter import VideoGetter
# from py360convert import e2c as getCubes
import multiprocessing.pool as pool
import torch
import numpy as np
from Utils.utils import (readFlo2Np, readImg2Np)
from Utils.utils import maprange
from torch.utils.data import Dataset, DataLoader
from Extras.loadconfigs import (TRAIN_LOADER_ARGS, 
                                TEST_LOADER_ARGS,
                                DATA_ROOT, 
                                IMG_TRANSFORM, 
                                FLO_TRANSFORM, 
                                DATA_ROOT,
                                S_LABELS,
                                M_LABELS,
                                LABELS)

import random

from tqdm import tqdm
from Utils.utils import stopExec

class EgoLoader(Dataset):
    def __init__(self, 
                 mode = 'train', 
                 shuffle = True, 
                 siamese = False, 
                 focus = False, 
                 flo = False, 
                 img = True, 
                 low = -1, 
                 high = 1, 
                 double_rots = False,
                 truncate = None
                 ):
        self.vd = VideoGetter(root = DATA_ROOT, mode = mode, shuffle = shuffle, truncate = truncate)
        self.siamese = siamese
        self.focus = focus
        self.flo = flo
        self.img = img
        self.R = lambda :np.random.uniform(low=low, high=high)
        self.rots = lambda :{'roll':self.R(), 'pitch':self.R(), 'yaw':self.R()}
        self.double_rots = double_rots
        self.img_transform = IMG_TRANSFORM
        self.flo_transform = FLO_TRANSFORM
    
    def __len__(self):
        return len(self.vd)
    
    def __getitem__(self, idx):
        data = self.vd[idx]
        
        sample = {'s_label':S_LABELS.get(data['s_label']),
                  'm_label':M_LABELS.get(data['m_label']),
                  'label':LABELS.get(data['label'])}
        
        if self.double_rots:
            sample['rots_1'] = self.rots()
        else:
            sample['rots_1'] = {'pitch':0, 'yaw':0, 'roll':0}
        
        if self.siamese:
            sample['rots_2'] = self.rots()
        if self.img:
            img_path_list = data['frame']
            imgs = list(map(readImg2Np, img_path_list))
            img = self.img_transform(np.concatenate(imgs,0))
            sample['img'] = img
            
        if self.flo:
            flo_path_list = data['flow']
            flos = list(map(readFlo2Np, flo_path_list))
            sample['flo'] = self.flo_transform(np.concatenate(flos,0))
                
        if self.focus:
            sample['focus'] = torch.from_numpy(np.load('focus.npy')[None,:]).float()
        return sample
    

def getEgoLoader(**kwargs):
    loader = EgoLoader(**kwargs)
    if kwargs.get('mode') == 'train':
        loader = DataLoader(loader, **TRAIN_LOADER_ARGS)
    else:
        loader = DataLoader(loader, **TEST_LOADER_ARGS)
    return loader

if __name__ == '__main__':
    loader = getEgoLoader(mode = 'train', 
                          shuffle = True, 
                          siamese = False, 
                          focus = False, 
                          flo = False, 
                          img = True, 
                          low = -1, 
                          high = 1, 
                          double_rots = False
                          )
    
    for _ in tqdm(loader):
        stopExec()
    