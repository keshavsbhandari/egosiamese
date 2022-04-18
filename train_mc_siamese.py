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
from Siamese.m_siamese import *
from Classifier.mc_siamese import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from Extras.loadconfigs import DEPTH,N_DEVICE, SERVER

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="Loss/val_loss_epoch",
    dirpath="cache",
    filename=f"{SERVER}_DEPTH_{DEPTH}_MOTION_SIAMESE_CLASSIFIER",
    save_top_k=1,
    mode="min",)
    
    # early_stop_callback = EarlyStopping(monitor="Loss/val_loss_epoch", 
    #                                     min_delta=1e-5, 
    #                                     patience=5, 
    #                                     verbose=True, 
    #                                     mode="min")
    
    trainer = Trainer(max_epochs = 1000, 
                      fast_dev_run = False, 
                      gpus = N_DEVICE,  
                      accelerator = "ddp", 
                      num_nodes = 1, 
                      callbacks=[checkpoint_callback],
                      # callbacks=[checkpoint_callback, early_stop_callback],
                      plugins=[DDPPlugin(find_unused_parameters=True)], 
                      reload_dataloaders_every_epoch=False,
                    #   log_every_n_steps = 1, 
                      # gradient_clip_val=1
                      )
    
    model = LTNClassifier()
    model.find_lr(1e-1, 1e-5)
    trainer.fit(model)