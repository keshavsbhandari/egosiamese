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
import math

from Model.model import *
from Utils.egoloader import *
from Utils.utils import *

from Extras.loadconfigs import (M_SIAMESE_PATH,
                                S_SIAMESE_PATH,
                                MC_SIAMESE_PATH,
                                SC_SIAMESE_PATH,
                                M_NUM_CLASSES,
                                S_NUM_CLASSES,
                                NUM_CLASSES)

"""
LIGHTNING LOGS : version_0
"""

class LTNSiamese(pl.LightningModule):
    def __init__(self, 
                 streamtype = 'flo', 
                 lr = 1e-4,
                 truncate_train = None,
                 truncate_val = None
                 ):
        super(LTNSiamese, self).__init__()
        assert streamtype in ['flo', 'img'], "streamtype must be either 'flo' or 'img'"
        self.streamtype = streamtype
        self.model = Siamese(Stream(in_channels = 2 if streamtype == 'flo' else 3))
        self.lr = lr
        self.truncate_train = truncate_train
        self.truncate_val = truncate_val
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            "optimizer":optimizer,
            "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95),
            "interval":"epoch",
            "monitor":"val_loss",
            "frequency":1,
        }
        
    def training_step(self, batch, batch_idx):
        stopExec()
        
        x = batch[self.streamtype]
        rots_1 = batch["rots_1"]
        rots_2 = batch["rots_2"]
        
        p1, p2, z1, z2 = self.model(x, rots_1 = rots_1, rots_2 = rots_2, mode = 'flo')
        loss = simLoss(p1,p2,z1,z2)
        
        self.log("Loss/train_loss", loss, prog_bar = True, on_step=True, on_epoch=True, sync_dist=True)
        return {'loss':loss, 'log': {'Loss/train': loss}}
    
    def validation_step(self, batch, batch_idx):
        stopExec()
        x = batch[self.streamtype]
        rots_1 = batch["rots_1"]
        rots_2 = batch["rots_2"]
        
        p1, p2, z1, z2 = self.model(x, rots_1 = rots_1, rots_2 = rots_2, mode = 'flo')
        loss = simLoss(p1,p2,z1,z2)
        
        self.log("Loss/val_loss", loss, prog_bar = True, on_step=True, on_epoch=True, sync_dist=True)
        return {'val_loss':loss, 'log': {'Loss/val': loss}}
    
    def find_lr(self, init_value, final_value):
        train_loader = self.train_dataloader()
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        optimizer = self.configure_optimizers().get('optimizer')
        optimizer.param_groups[0]["lr"] = lr
        best_loss = -1.0
        batch_num = 0
        losses = []
        log_lrs = []
        iterator = tqdm(train_loader, desc="Current lr=XX.XX Steps=XX Loss=XX.XX Best lr=XX.XX ")
        for i, batch in enumerate(iterator):
            batch_num += 1
            x = batch[self.streamtype]
            rots_1 = batch["rots_1"]
            rots_2 = batch["rots_2"]
            optimizer.zero_grad()
            p1, p2, z1, z2 = self.model(x, rots_1 = rots_1, rots_2 = rots_2, mode = 'flo')
            loss = simLoss(p1,p2,z1,z2)

            # Crash out if loss explodes
            # if batch_num > 1 and loss > 4 * best_loss:
            #     print("CRASH OUT")
            #     return log_lrs[10:-5], losses[10:-5]
            # Record the best loss
            if loss < best_loss or batch_num == 1:
                best_loss = loss
                best_lr = lr
            # Do the backward pass and optimize
            loss.backward()
            optimizer.step()
            iterator.set_description("Current lr=%5.9f Steps=%d Loss=%5.3f Best lr=%5.9f " %(lr, i, loss, best_lr))
            # Store the values
            losses.append(loss.detach())
            log_lrs.append(math.log10(lr))
            # Update the lr for the next step and store
            lr = lr * update_step
            optimizer.param_groups[0]["lr"] = lr
        logs, losses = log_lrs[10:-5], losses[10:-5]

        plt.plot(logs, losses)
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.savefig(f"{self.streamtype} - Optimal lr curve.png")
        print("plot saved")
    
    def train_dataloader(self):
        loader = getEgoLoader(mode = 'train', 
                              shuffle = True, 
                              siamese = True, 
                              focus = False, 
                              flo = True, 
                              img = False, 
                              low = -1, 
                              high = 1, 
                              double_rots = False,
                              truncate = self.truncate_train)
        return loader
        
    def val_dataloader(self):
        loader = getEgoLoader(mode = 'test', 
                              shuffle = True, 
                              siamese = True, 
                              focus = False, 
                              flo = True, 
                              img = False, 
                              low = -1, 
                              high = 1, 
                              double_rots = False,
                              truncate = self.truncate_val)
        return loader

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="cache",
    filename="SIAMESE",
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
    
    model = LTNSiamese()
    # model = LTNSiamese.load_from_checkpoint("cache/SIAMESE.ckpt")
    print(model.model)
    trainer.fit(model)