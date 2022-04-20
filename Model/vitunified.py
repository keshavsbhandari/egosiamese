import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
from equilib import equi2cube as getCubes
from Utils.utils import maprange
from functorch import vmap

from Extras.loadconfigs import NUM_CLASSES, DEPTH, EMB_DIM, TANGENT_PATCH, FEATURE_AGGREGATOR, ENC_SEQ_DIM

from Model.model import *

class UnifiedStream(nn.Module):
    def __init__(self,
                 nhead = 64,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 batch_first = True,
                 layer_norm_eps = 1e-5
                 ):
        super(UnifiedStream, self).__init__()
        self.d_model = EMB_DIM
        
        self.patcherRGB = PerspectivePatch(in_channels=3)
        self.patcherFLO = PerspectivePatch(in_channels=2)
        
        self.patchFeaturesRGB = PatchWiseFeatureAggregator(in_channels=3, depth=DEPTH)
        self.patchFeaturesFLO = PatchWiseFeatureAggregator(in_channels=2, depth=DEPTH)
        
        self.posEncoding1 = PositionalEncoding(d_model=ENC_SEQ_DIM*2)
        self.posEncoding2 = PositionalEncoding(d_model=EMB_DIM*2)
        
        self.featureAggr = TransformerEncoderLayer(d_model = ENC_SEQ_DIM*2, 
                                           nhead = nhead//2, 
                                           dim_feedforward=dim_feedforward//4, 
                                           dropout=dropout, 
                                           layer_norm_eps=layer_norm_eps, 
                                           batch_first=batch_first,
                                           )
        
        self.encoder_1 = TransformerEncoderLayer(d_model = self.d_model * 2, 
                                           nhead = nhead, 
                                           dim_feedforward=dim_feedforward, 
                                           dropout=dropout, 
                                           layer_norm_eps=layer_norm_eps, 
                                           batch_first=batch_first,
                                           )
        
        self.encoder_2 = TransformerEncoderLayer(d_model = self.d_model * 2, 
                                           nhead = nhead, 
                                           dim_feedforward=dim_feedforward, 
                                           dropout=dropout, 
                                           layer_norm_eps=layer_norm_eps, 
                                           batch_first=batch_first,
                                           )
        
        self.encoder_3 = TransformerEncoderLayer(d_model = self.d_model * 2, 
                                           nhead = nhead, 
                                           dim_feedforward=dim_feedforward, 
                                           dropout=dropout, 
                                           layer_norm_eps=layer_norm_eps, 
                                           batch_first=batch_first,
                                           )
        
        self.fc = nn.Linear(in_features = 6*2*self.d_model, out_features=self.d_model)
        
    def forward(self, rgb, flo, **kwargs):
        # rgb
        rots_1 = flattenRots(kwargs.get('rots_1'))
        rgb = getCubes(rgb,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
        B,_,H,W = rgb.shape
        rgb = rgb.view(B,-1,3,H,W).permute(0,2,1,3,4)
        
        #flo
        rmaps, flo = maprange(flo)
        flo = getCubes(flo,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
        _, flo = maprange(flo, **rmaps)
        B,_,H,W = flo.shape
        flo = flo.view(B,-1,2,H,W).permute(0,2,1,3,4)
        
        
        # main forward
        rgb1 = vmap(self.patchFeaturesRGB, in_dims = 4)(rgb.unfold(4, rgb.size(3),rgb.size(3))).permute(1,2,0)
        flo1 = vmap(self.patchFeaturesFLO, in_dims = 4)(flo.unfold(4, flo.size(3),flo.size(3))).permute(1,2,0)
        
        
        rgb2 = self.patcherRGB(rgb)
        flo2 = self.patcherFLO(flo)
        
        x1 = torch.cat((rgb1, flo1), 1)#256,6
        x2 = torch.cat((rgb2, flo2), -1)#6,768*2
        
        
        x2 = torch.einsum('ijk,ikm->ijm', x1, x2)#256,768*2
        x2 = nn.LeakyReLU()(x2)
        
        x1 = self.posEncoding1(x1.permute(0,2,1)).permute(0,2,1)#256,6
        x2 = self.posEncoding2(x2)#256,768*2
        
        x2 = self.encoder_1(x2)#256,768*2
        x2 = self.encoder_2(x2)#256,768*2
        
        x = torch.einsum('ijk,ijm->ikm',x1,x2)
        x = nn.LeakyReLU()(x)
        
        x = self.encoder_3(x)
        x = self.fc(x.view(x.size(0), -1))
        
        if kwargs.get('softmax'):
            return F.softmax(x, dim = 1)
        
        return x
        
        
        
        
        
        
        
        