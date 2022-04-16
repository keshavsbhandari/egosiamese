from __future__ import print_function, division
from secrets import choice
from torch.utils.data import Dataset, DataLoader
from Utils.datacollector import DataCollector

from Extras.loadconfigs import (M_NUM_CLASSES,
                                S_NUM_CLASSES,
                                NUM_CLASSES)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SpatialMotionLoader(Dataset):
    def __init__(self, 
                 root_dir = "/data/keshav/360/finalEgok360/data/", 
                 mode = 'train', 
                 transform = None, 
                 img = True, 
                 flo = False,
                 siamese = False,
                 ignoreFocus = True
                 ):
        assert img or flo, "Either both or one from [flo, img] should be True"
        self.siamese = siamese
        self.s_class = S_NUM_CLASSES
        self.m_class = M_NUM_CLASSES
        self.sm_class = NUM_CLASSES
        
        self.img = img
        self.flo = flo
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.datalist = range(10240)
        
        self.collector = DataCollector(root = root_dir, 
                                       mode = mode, 
                                       shuffle = mode =='train', 
                                       siamese = siamese,
                                       getimg=img,
                                       getflo=flo,
                                       ignoreFocus=ignoreFocus)
    
    def __len__(self):
        return len(self.collector)
    
    def __getitem__(self, index):
        sample = self.collector[index]
        return sample


def getLoader(root_dir = "/data/keshav/360/finalEgok360/data/",
              mode = 'train',
              batch_size = 1,
              shuffle = False,
              num_workers = 0,
              img = True,
              flo = True,
              siamese = True,
              ignoreFocus = False,
              pin_memory = False,
              prefetch_factor=2,
              persistent_workers = False,
              drop_last = True
            ):
    return DataLoader(SpatialMotionLoader(root_dir, 
                                          mode = mode, 
                                          img = img, 
                                          flo = flo,
                                          siamese = siamese,
                                          ignoreFocus=ignoreFocus
                                          ), 
                      batch_size = batch_size,
                      shuffle = shuffle,
                      num_workers = num_workers,
                      pin_memory=pin_memory,
                      prefetch_factor=prefetch_factor,
                      persistent_workers = persistent_workers,
                      drop_last = drop_last
                      )

if __name__ == '__main__':
    dt = getLoader(root_dir = None, img = True, flo = True)
    sample = next(iter(dt))
    for s in sample:
        print(s, sample[s].shape)