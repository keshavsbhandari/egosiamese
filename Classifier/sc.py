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

    def find_lr(self, init_value, final_value):
        train_loader = getLoader(**self.train_loader_args)
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        optimizer = self.configure_optimizers()
        optimizer.param_groups[0]["lr"] = lr
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        iterator = tqdm(train_loader, desc="Current lr=XX.XX Steps=XX Loss=XX.XX Best lr=XX.XX ")
        for i, data in enumerate(iterator):
            batch_num += 1
            inputs, labels = data[f"{self.streamtype}_1"], data[f"{self.streamtype}_label"]
            optimizer.zero_grad()
            prob, _ = self.model(inputs, softmax=True)
            loss = torch.nn.CrossEntropyLoss()(prob, labels)
            # Crash out if loss explodes
            if batch_num > 1 and loss > 4 * best_loss:
                return log_lrs[10:-5], losses[10:-5]
            # Record the best loss
            if loss < best_loss or batch_num == 1:
                best_loss = loss
                best_lr = lr
            # Do the backward pass and optimize
            loss.backward()
            self.optimizer.step()
            iterator.set_description("Current lr=%5.9f Steps=%d Loss=%5.3f Best lr=%5.9f " %(lr, i, loss, best_lr))
            # Store the values
            losses.append(loss.detach())
            log_lrs.append(math.log10(lr))
            # Update the lr for the next step and store
            lr = lr * update_step
            self.optimizer.param_groups[0]["lr"] = lr
        logs, losses = log_lrs[10:-5], losses[10:-5]

        plt.plot(logs, losses)
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.savefig("Optimal lr curve.png")
        print("plot saved")

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