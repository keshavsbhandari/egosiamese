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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from Model.model import *
from Utils.dataloader import *
from Utils.utils import *
from Siamese.m_siamese import *
from Classifier.sc_siamese import *
from Extras.loadconfigs import DEPTH

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="Loss/val_loss_epoch",
    dirpath="cache",
    filename=f"DEPTH_{DEPTH}_SPATIAL_SIAMESE_CLASSIFIER",
    save_top_k=1,
    mode="min",)
    
    # early_stop_callback = EarlyStopping(monitor="Loss/val_loss_epoch", 
    #                                     min_delta=1e-5, 
    #                                     patience=5, 
    #                                     verbose=True, 
    #                                     mode="min")
    
    trainer = Trainer(max_epochs = 1000, 
                      fast_dev_run = False, 
                      gpus = 8,  
                      accelerator = "ddp", 
                      num_nodes = 1, 
                      # callbacks=[checkpoint_callback,early_stop_callback],
                      callbacks=[checkpoint_callback],
                      plugins=[DDPPlugin(find_unused_parameters=True)], 
                      reload_dataloaders_every_epoch=False,
                    #   log_every_n_steps = 1, 
                    #   gradient_clip_val=0.5
                      )
    
    model = LTNClassifier()
    model = model.load_from_checkpoint(f"cache/DEPTH_{DEPTH}_SPATIAL_SIAMESE_CLASSIFIER-v1.ckpt")
    trainer.fit(model)
    # trainer.validate(model)
    