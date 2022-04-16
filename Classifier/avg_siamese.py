import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Model.model import *
from Utils.dataloader import *
from Utils.utils import *
from Siamese.s_siamese import *
from Classifier.mc_siamese import LTNClassifier as MClassifier
from Classifier.sc_siamese import LTNClassifier as SClassifier

from Extras.loadconfigs import (MC_SIAMESE_PATH,
                                SC_SIAMESE_PATH,
                                NUM_CLASSES)

class AvgClassifier(pl.LightningModule):
    def __init__(self, 
                 lr = 1e-3,
                 train_loader_args = dict(batch_size = 16,
                                          shuffle = True,
                                          num_workers = 4,
                                          mode = 'train',
                                          flo = True,
                                          img = True,
                                          siamese = False
                                          ),
                 val_loader_args = dict(batch_size = 16,
                                          shuffle = False,
                                          num_workers = 4,
                                          mode = 'val',
                                          flo = True,
                                          img = True,
                                          siamese = False
                                          ),
                 ):
        super(AvgClassifier, self).__init__()
        
        self.train_loader_args = train_loader_args
        self.val_loader_args = val_loader_args
        self.model = Fuser(spatial_classifier = SClassifier.load_from_checkpoint(SC_SIAMESE_PATH).model,
                           motion_classifier  =  MClassifier.load_from_checkpoint(MC_SIAMESE_PATH).model,
                           num_classes=NUM_CLASSES)
        self.lr = lr
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return {
            "optimizer":optimizer,
            "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95),
            "interval":"epoch",
            "monitor":"val_loss",
            "frequency":1,
        }
        
    def training_step(self, batch, batch_idx):
        stopExec()
        img, flo, y = batch["img_1"], batch["flo_1"], batch["label"]
        prob = self.model(x_spatial = img,
                             x_motion = flo, 
                             softmax=True)
        loss = torch.nn.CrossEntropyLoss()(prob, y)
        
        self.log("train_loss", loss, prog_bar = True)
        
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        stopExec()
        img, flo, y = batch["img_1"], batch["flo_1"], batch["label"]
        prob = self.model(x_spatial = img,
                             x_motion = flo, 
                             softmax=True)
        loss = torch.nn.CrossEntropyLoss()(prob, y)
        
        self.log("val_loss", loss, prog_bar = True)
        
        return {'val_loss':loss}
    
    def train_dataloader(self):
        loader = getLoader(**self.train_loader_args)
        return loader
        
    def val_dataloader(self):
        loader = getLoader(**self.val_loader_args)
        return loader

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="cache",
    filename="AVG_SIAMESE_CLASSIFIER",
    save_top_k=1,
    mode="min",)
    
    trainer = Trainer(max_epochs = 1000, 
                      fast_dev_run = False, 
                      gpus = 8,  
                      accelerator = "ddp", 
                      num_nodes = 1, 
                      callbacks=[checkpoint_callback],
                      plugins=[DDPPlugin(find_unused_parameters=True)], 
                      log_every_n_steps = 1, 
                      gradient_clip_val=0.5
                      )
    
    model = AvgClassifier(sclassifier_path = 'cache/SPATIAL_SIAMESE_CLASSIFIER.ckpt',
                          mclassifier_path = 'cache/MOTION_SIAMESE_CLASSIFIER.ckpt',
                          num_classes = 16,
                          lr = 1e-3,
                          train_loader_args = dict(batch_size = 16,
                                          shuffle = True,
                                          num_workers = 4,
                                          mode = 'train',
                                          flo = True,
                                          img = True,
                                          siamese = False
                                          ),
                          val_loader_args = dict(batch_size = 16,
                                          shuffle = False,
                                          num_workers = 4,
                                          mode = 'val',
                                          flo = True,
                                          img = True,
                                          siamese = False
                                          ),
                          )
    trainer.fit(model)