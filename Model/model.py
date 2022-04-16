import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
from equilib import equi2cube as getCubes
from Utils.utils import maprange

from Extras.loadconfigs import NUM_CLASSES, DEPTH, EMB_DIM, TANGENT_PATCH

def flattenRots(rots):
    for r,value in rots.items():
        rots[r] = value.detach().cpu().numpy().tolist()
    return pd.DataFrame(rots).to_dict(orient='records')

class PerspectivePatch(nn.Module):
    def __init__(self, in_channels = 3):
        super(PerspectivePatch, self).__init__()
        if DEPTH:    
            self.patcher = nn.Conv3d(in_channels = in_channels,
                                out_channels = EMB_DIM,
                                kernel_size = (DEPTH, TANGENT_PATCH, TANGENT_PATCH),
                                stride = (DEPTH, TANGENT_PATCH, TANGENT_PATCH),
                                )
            self.norm = torch.nn.BatchNorm3d(EMB_DIM, eps=1e-05, momentum=0.1, affine=True)
        else:
            self.patcher = nn.Conv2d(in_channels = in_channels,
                                out_channels = EMB_DIM,
                                kernel_size = TANGENT_PATCH,
                                stride = TANGENT_PATCH, 
                                )
            self.norm = torch.nn.BatchNorm2d(EMB_DIM, eps=1e-05, momentum=0.1, affine=True)
        
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.patcher(x)
        x = self.norm(x)
        x = self.act(x)
        x = x.view(x.size(0), x.size(1),-1).permute(0,2,1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)
        return self.dropout(x)
        
class Stream(nn.Module):
    def __init__(self,
               in_channels = 2,
               nhead = 64, 
               dim_feedforward = 2048, 
               dropout = 0.1, 
               batch_first = True, 
               layer_norm_eps = 1e-05
               ):
        super(Stream, self).__init__()
        self.d_model = EMB_DIM
        self.posEncoding = PositionalEncoding(d_model=EMB_DIM)
        self.patcher = PerspectivePatch(in_channels=in_channels)
        self.encoder_1 = TransformerEncoderLayer(d_model = EMB_DIM, 
                                           nhead = nhead, 
                                           dim_feedforward=dim_feedforward, 
                                           dropout=dropout, 
                                           layer_norm_eps=layer_norm_eps, 
                                           batch_first=batch_first,
                                           )
        self.encoder_2 = TransformerEncoderLayer(d_model = EMB_DIM, 
                                           nhead = nhead//2, 
                                           dim_feedforward=dim_feedforward//4, 
                                           dropout=dropout, 
                                           layer_norm_eps=layer_norm_eps, 
                                           batch_first=batch_first,
                                           )
        self.fc = nn.Linear(in_features = 6*EMB_DIM, out_features=EMB_DIM)
    
    def forward(self, x, **kwargs):
        if kwargs.get('rots_1'):
            # rots_1 = pd.DataFrame(kwargs.get('rots_1')).to_dict(orient='records')
            rots_1 = flattenRots(kwargs.get('rots_1'))
            if kwargs['mode'] == 'flo':
                    rmaps, x = maprange(x)
                    x = getCubes(x,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
                    _, x = maprange(x, **rmaps)
                    
                    B,_,H,W = x.shape
                    x = x.view(B,-1,2,H,W).permute(0,2,1,3,4)
                    
            elif kwargs['mode'] == 'img':
                x = getCubes(x,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
                B,_,H,W = x.shape
                x = x.view(B,-1,3,H,W).permute(0,2,1,3,4)
            else:
                raise Exception(f"Mode must be in [flo,img], but provided {kwargs['mode']}")
        
        x = self.patcher(x)
        x = self.posEncoding(x)
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Siamese(nn.Module):
    def __init__(self, base_encoder, pred_dim = 128):
        super(Siamese, self).__init__()
        self.encoder = base_encoder
        # build 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True), # first layer
                            nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(inplace=True), # second layer
                            self.encoder.fc,
                            nn.BatchNorm1d(EMB_DIM, affine=False)) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
        # build 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(EMB_DIM, pred_dim, bias = False),
                                   nn.BatchNorm1d(pred_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(pred_dim, EMB_DIM))
    def forward(self, x, **kwargs):
        # PRE-PROCESSING STARTED | INSIDE FORWARD FOR CUDA ACCELERATION
        if kwargs.get('rots_2'):
            # rots_1 = pd.DataFrame(kwargs.get('rots_1')).to_dict(orient='records')
            # rots_2 = pd.DataFrame(kwargs.get('rots_2')).to_dict(orient='records')
            
            rots_1 = flattenRots(kwargs.get('rots_1'))
            rots_2 = flattenRots(kwargs.get('rots_2'))
            
            
            if kwargs['mode'] == 'flo':
                rmaps, x = maprange(x)
                x1 = getCubes(x,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
                x2 = getCubes(x,w_face=TANGENT_PATCH, rots = rots_2, cube_format='horizon')
                _, x1 = maprange(x1, **rmaps)
                _, x2 = maprange(x2, **rmaps)
                
                B,_,H,W = x1.shape
                x1 = x1.view(B,-1,2,H,W).permute(0,2,1,3,4)
                x2 = x2.view(B,-1,2,H,W).permute(0,2,1,3,4)
                
            elif kwargs['mode'] == 'img':
                x1 = getCubes(x,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
                x2 = getCubes(x,w_face=TANGENT_PATCH, rots = rots_2, cube_format='horizon')
                
                B,_,H,W = x1.shape
                x1 = x1.view(B,-1,3,H,W).permute(0,2,1,3,4)
                x2 = x2.view(B,-1,3,H,W).permute(0,2,1,3,4)
                
            else:
                raise Exception(f"Mode must be in [flo,img], but provided {kwargs['mode']}")
        else:
            # rots_1 = pd.DataFrame(kwargs.get('rots_1')).to_dict(orient='records')
            rots_1 = flattenRots(kwargs.get('rots_1'))
            if kwargs['mode'] == 'flo':
                rmaps, x = maprange(x)
                x = getCubes(x,w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
                _, x = maprange(x, **rmaps)
                
                B,_,H,W = x.shape
                x = x.view(B,-1,2,H,W).permute(0,2,1,3,4)
                
            elif kwargs['mode'] == 'img':
                x = getCubes(x, w_face=TANGENT_PATCH, rots = rots_1, cube_format='horizon')
                
                B,_,H,W = x.shape
                x = x.view(B,-1,3,H,W).permute(0,2,1,3,4)
                
            else:
                raise Exception(f"Mode must be in [flo,img], but provided {kwargs['mode']}")
            return self.encoder(x)
        # PRE-PROCESSING ENDED | INSIDE FORWARD FOR CUDA ACCELERATION
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


class Classifier(nn.Module):
    def __init__(self,
               stream,
               num_classes = NUM_CLASSES):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.stream = stream
        try:    
            self.fc = nn.Linear(in_features=self.stream.encoder.d_model, 
                                out_features=num_classes)
        except:
            self.fc = nn.Linear(in_features=self.stream.d_model, 
                                out_features=num_classes)
    def forward(self,x, softmax = False, **kwargs):
        z = self.stream(x, **kwargs)
        x = self.fc(z)
        if softmax:
            x = F.softmax(x, dim = 1)
        return x, z

class Fuser(nn.Module):
    def __init__(self, 
               spatial_classifier, 
               motion_classifier,
               num_classes = NUM_CLASSES,
               mode = 'avg'
               ):
        super(Fuser, self).__init__()
        assert mode in ['avg','concat'], "mode should be either from [avg, concat]"
        self.motion_classifier = motion_classifier
        self.spatial_classifier = spatial_classifier
        self.mode = mode
        if mode == 'avg':
            try:    
                in_features = self.motion_classifier.stream.encoder.d_model
            except:
                in_features = self.motion_classifier.stream.d_model
        else:
            in_features = self.motion_classifier.num_classes + self.spatial_classifier.num_classes
        
        self.linear = nn.Linear(in_features = in_features, 
                                out_features = num_classes,
                                bias = False)
    def forward(self, x_spatial, x_motion, softmax = False, **kwargs):
        if self.mode == 'avg':  
            _, spatial = self.spatial_classifier(x_spatial, mode = 'img', **kwargs)
            _, motion = self.motion_classifier(x_motion, mode = 'flo', **kwargs)
        else:
            spatial, _ = self.spatial_classifier(x_spatial, mode = 'img', **kwargs)
            motion, _ = self.motion_classifier(x_motion, mode = 'flo', **kwargs)
        
        if self.mode == 'avg':  
            x = (spatial + motion)/2.0
        else:
            x = torch.cat((spatial,motion),-1)
        x = self.linear(x)
        if softmax:
            x = F.softmax(x, dim = 1)
        return x
    
    
    