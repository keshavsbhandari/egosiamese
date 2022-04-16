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

from Extras.loadconfigs import S_NUM_CLASSES

class LTNClassifier(pl.LightningModule):
    def __init__(self,
                 streamtype = 'img',
                 lr = 1e-3,
                 train_loader_args = dict(batch_size = 16,
                                          shuffle = True,
                                          num_workers = 4,
                                          mode = 'train',
                                          flo = False,
                                          img = True,
                                          siamese = False
                                          ),
                 val_loader_args = dict(batch_size = 16,
                                          shuffle = False,
                                          num_workers = 4,
                                          mode = 'val',
                                          flo = False,
                                          img = True,
                                          siamese = False
                                          ),
                 ):
        super(LTNClassifier, self).__init__()
        assert streamtype in ['flo', 'img'], "streamtype must be either 'flo' or 'img'"
        
        self.train_loader_args = train_loader_args
        self.val_loader_args = val_loader_args
        
        self.streamtype = streamtype
        
        self.model = Classifier(Stream(in_channels=3), num_classes=S_NUM_CLASSES)
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
        x, y = batch[f"{self.streamtype}_1"], batch[f"{self.streamtype}_label"]
        prob, _ = self.model(x, softmax=True)
        loss = torch.nn.CrossEntropyLoss()(prob, y)
        
        self.log("train_loss", loss, prog_bar = True)
        
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        stopExec()
        x, y = batch[f"{self.streamtype}_1"], batch[f"{self.streamtype}_label"]
        prob, _ = self.model(x, softmax=True)
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
    filename="SPATIAL_CLASSIFIER",
    save_top_k=1,
    mode="min",)
    
    trainer = Trainer(max_epochs = 1000, 
                      fast_dev_run = False, 
                      gpus = 8,  
                      accelerator = "ddp", 
                      num_nodes = 1, 
                      callbacks=[checkpoint_callback],
                      plugins=[DDPPlugin(find_unused_parameters=False)], 
                      log_every_n_steps = 1, 
                      gradient_clip_val=0.5
                      )
    
    model = LTNClassifier()
    trainer.fit(model)