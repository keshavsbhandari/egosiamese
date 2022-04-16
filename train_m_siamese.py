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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from Siamese.m_siamese import *
from Extras.loadconfigs import DEPTH

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="Loss/val_loss_epoch",
    dirpath="cache",
    filename=f"DEPTH_{DEPTH}_MOTION_SIAMESE",
    save_top_k=1,
    mode="min",)
    
    early_stop_callback = EarlyStopping(monitor="Loss/val_loss_epoch", 
                                        min_delta=1e-3, 
                                        patience=3, 
                                        verbose=True, 
                                        mode="min")
    
    trainer = Trainer(max_epochs = 1000, 
                      fast_dev_run = False, 
                      gpus = 8,  
                      accelerator = "ddp",
                      num_nodes = 1, 
                      callbacks=[checkpoint_callback, early_stop_callback],
                      plugins=[DDPPlugin(find_unused_parameters=False)],
                      reload_dataloaders_every_epoch=False,
                      # log_every_n_steps = 1, 
                      # gradient_clip_val=0.5
                      )
    
    model = LTNSiamese()
    # model.load_from_checkpoint("cache/DEPTH_DEPTH=0_MOTION_SIAMESE-BEST.ckpt")
    trainer.fit(model)
    # trainer.validate(model)