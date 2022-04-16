from Model.model import *
from Utils.egoloader import getEgoLoader
from Utils.utils import count_parameters
import torch
from Extras.loadconfigs import (DEPTH, 
                                PATCH_W, 
                                PATCH_H,
                                M_NUM_CLASSES, 
                                S_NUM_CLASSES, 
                                NUM_CLASSES)

if __name__ == '__main__':
    # SANITY CHECK
    loader = getEgoLoader(mode = 'train', 
                          shuffle = True, 
                          siamese = True, 
                          focus = False, 
                          flo = True, 
                          img = True, 
                          low = -1, 
                          high = 1, 
                          double_rots = False)
    
    sample = next(iter(loader))
    
    # if DEPTH:    
    #     f = torch.rand(8, 2, DEPTH, PATCH_H, PATCH_W).float()
    #     i = torch.rand(8, 3, DEPTH, PATCH_H, PATCH_W).float()
    # else:
    #     f = sample.get('flo')
    #     i = sample.get('img')
    
    f = sample.get('flo')
    i = sample.get('img')
    
    r1 = sample.get('rots_1')
    r2 = sample.get('rots_2')
    
    # print(f.shape)
    # print(i.shape)
    
    
    motion = Stream()
    spatia = Stream(in_channels=3)
    
    print(f"Parameters Count\nmotion: {count_parameters(motion)}\nspatia: {count_parameters(spatia)}")
    
    mSiamese = Siamese(motion)
    sSiamese = Siamese(spatia)
    
    print(f"Parameters Count\nmSiamese: {count_parameters(mSiamese)}\nsSiamese: {count_parameters(sSiamese)}")
    
    mClassifier = Classifier(mSiamese, num_classes = M_NUM_CLASSES)
    sClassifier = Classifier(sSiamese, num_classes = S_NUM_CLASSES)
    
    print(f"Parameters Count\nmClassifier: {count_parameters(mClassifier)}\nsClassifier: {count_parameters(sClassifier)}")
    
    avgFuse = Fuser(sClassifier, mClassifier, mode = 'avg', num_classes=NUM_CLASSES)
    catFuse = Fuser(sClassifier, mClassifier, mode = 'concat', num_classes=NUM_CLASSES)
    
    print(f"Parameters Count\navgFuse: {count_parameters(avgFuse)}\ncatFuse: {count_parameters(catFuse)}")
    
    
    
    print(f"Shape\nflo: {f.shape}\nimg: {i.shape}")
    
    f_motion = motion(f, rots_1 = r1, mode = 'flo')
    i_spatia = spatia(i, rots_1 = r1, mode = 'img')
    
    print(f"Shape\nf_motion: {f_motion.shape}\ni_spatia: {i_spatia.shape}")
    
    m_p1, m_p2, m_z1, m_z2 = mSiamese(f, rots_1 = r1, rots_2 = r2, mode = 'flo')
    i_p1, i_p2, i_z1, i_z2 = sSiamese(i, rots_1 = r1, rots_2 = r2, mode = 'img')
    
    
    print(f"Shape\nm_siamese(p1,p2,z1,z2): {m_p1.shape, m_p2.shape, m_z1.shape, m_z2.shape}\ni_siamese(p1,p2,z1,z2): {i_p1.shape, i_p2.shape, i_z1.shape, i_z2.shape}")
    
    x_mClass, z_mClass = mClassifier(f, rots_1 = r1, mode = 'flo')
    x_sClass, z_sClass = sClassifier(i, rots_1 = r1, mode = 'img')
    
    print(f"Shape\n(x_mClass,z_mClass): {x_mClass.shape, z_mClass.shape}\n(x_sClass,z_sClass): {x_sClass.shape, z_sClass.shape}")
    
    x_avg = avgFuse(i, f, rots_1 = r1)
    
    print(f"shape\nx_avg: {x_avg.shape}")
    
    x_cat = catFuse(i, f, rots_1 = r1)
    
    print(f"shape\nx_cat: {x_cat.shape}")