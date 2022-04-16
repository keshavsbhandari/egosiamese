
from Utils.egogetter import VideoGetter
import Augmentor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random
import torch
from Extras.loadconfigs import EMB_DIM, S_LABELS, M_LABELS, LABELS, TANGENT_PATCH, DEPTH
import albumentations as A
from Utils.utils import getPanoRowsImg,getPanoRowsFlow
import equilib as eq
from imageio import imread

class DataCollector(object):
    def __init__(self, 
                 root = "/data/keshav/360/finalEgok360/data/",
                 mode = "test",
                 shuffle = True,
                 w = 640, 
                 h = 320,
                 siamese=False, 
                 ignoreFocus = False,
                 getimg = True,
                 getflo = True):
        self.getimg = getimg
        self.getflo = getflo
        self.n = DEPTH
        self.w = w
        self.h = h
        self.ignoreFocus = ignoreFocus
        self.siamese = siamese
        self.aug = self.albAugmentor(DEPTH)
        self.actMapper = self.getMapGenerator()
        self.data = VideoGetter(root = root, mode = mode, nwor = DEPTH, shuffle = shuffle)
        
        self.imgTransform = T.Compose([T.ToTensor(), 
                                       T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225],
                                                  ),
                                      ])
        self.floTransform = T.Compose([T.Normalize(mean=[0.485, 0.456], 
                                                   std=[0.229, 0.224],
                                                  ),
                                      ])
    
    def albAugmentor(self, num_imgs):
        aug = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8),
                         A.RandomFog(fog_coef_upper=0.5,p = 0.2),
                        ],
                        additional_targets={f'image{i}': 'image' for i in range(num_imgs - 1)}
                       )
        return aug
    
    def mapAugMentation(self, img_list, aug):
        img_list = list(map(np.asarray,map(Image.open, img_list)))
        keys = ['image']+[f"image{i}" for i in range(len(img_list)-1)]
        augargs = dict(zip(keys, img_list))
        out = aug(**augargs)
        aug_img = [Image.fromarray(out[k]) for k in keys]
        return aug_img
    
    def focusGenerator(self, w = 640, h = 320):
        u,v = np.meshgrid(np.linspace(1,-1,w), np.linspace(1,-1,h))
        radi = (u**2 + v**2)**0.5
        radi = radi/radi.max()
        radi = abs(radi -1 )
        radi = torch.from_numpy(radi)[None,:]
        radi = eq.equi2equi(radi, rots = {'pitch':-np.pi/2, 'yaw':0, 'roll':0})
        return radi
    
    def getMapGenerator(self):
        con = torch.nn.Conv2d(in_channels=1, 
                              out_channels=EMB_DIM, 
                              kernel_size=64, 
                              stride=64, 
                              bias=False).requires_grad_(False)
        con.weight.data = torch.ones_like(con.weight.data).type(con.weight.data.dtype)
        return con
    
    def mapFocusWithConv(self, mapper, focus):
        overlay = mapper(focus.unsqueeze(0))
        overlay = overlay.view(1,-1,6).permute(0,2,1)
        return overlay
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        frames = data['frame']
        flows = data['flow']
        
        if self.getflo:    
            flow_list = [np.load(f) for f in flows]
        
        img_list = self.mapAugMentation(frames, self.aug)
        rotargs = {'pitch':None, 'yaw':None, 'roll':None}
        
        sample = {}
        
        img_list_1 = []
        flow_list_1 = []
        if self.getimg:    
            for i, img in enumerate(img_list):
                temp, rotargs = getPanoRowsImg(img = img, face_w = TANGENT_PATCH, return_rot_args = True, **rotargs)
                img_list_1.append(temp)
        if self.getflo:    
            for i, flow in enumerate(flow_list):
                temp, rotargs = getPanoRowsFlow(flow = flow, face_w = TANGENT_PATCH, return_rot_args = True, **rotargs)
                flow_list_1.append(temp)
        if not self.ignoreFocus:
            focus_1 = eq.equi2equi(self.focusGenerator(w=self.w, h=self.h), rots=rotargs)
            focus_1 = self.actMapper(focus_1[None,:].float())
        
        if self.siamese:
            img_list_2 = []
            flow_list_2 = []
            
            rotargs = {'pitch':None, 'yaw':None, 'roll':None}
            if self.getimg:    
                for i, img in enumerate(img_list):
                    temp, rotargs = getPanoRowsImg(img = img, face_w = TANGENT_PATCH, return_rot_args = True, **rotargs)
                    img_list_2.append(temp)
            if self.getflo:    
                for i, flow in enumerate(flow_list):
                    temp, rotargs = getPanoRowsFlow(flow = flow, face_w = TANGENT_PATCH, return_rot_args = True, **rotargs)
                    flow_list_2.append(temp)
            
            if not self.ignoreFocus:
                focus_2 = eq.equi2equi(self.focusGenerator(w=self.w, h=self.h), rots=rotargs)
                focus_2 = self.actMapper(focus_2[None,:].float())
            if self.getimg:    
                imgs_2 = list(map(self.imgTransform, img_list_2))
                imgs_2 = torch.stack(imgs_2, 1)
            if self.getflo:    
                flows_2 = [self.floTransform(torch.from_numpy(f).permute(2,0,1)) for f in flow_list_2]
                flows_2 = torch.stack(flows_2, 1)
            if self.getimg:    
                sample['img_2'] = imgs_2
            if self.getflo:    
                sample['flo_2'] = flows_2
            if not self.ignoreFocus:
                sample['focus_2'] = focus_2
        
        if self.getimg:    
            imgs_1 = list(map(self.imgTransform, img_list_1))
            imgs_1 = torch.stack(imgs_1, 1)
        
        if self.getflo:    
            flows_1 = [self.floTransform(torch.from_numpy(f).permute(2,0,1)) for f in flow_list_1]
            flows_1 = torch.stack(flows_1, 1)
        
        if self.getimg:    
            sample['img_1'] = imgs_1
        if self.getflo:    
            sample['flo_1'] = flows_1
        
        if not self.ignoreFocus:
            sample['focus_1'] = focus_1
        
        sample['s_label'] = S_LABELS[data['s_label']]
        sample['m_label'] = M_LABELS[data['m_label']]
        sample['label'] = LABELS[data['label']]
        
        return sample