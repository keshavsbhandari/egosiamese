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
from Utils.utils import skipZerothDeviceInPanda

from Model.model import *
from Utils.dataloader import *
from Utils.utils import *
from Siamese.s_siamese import *
from Classifier.avg_siamese import *

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
    
    model = AvgClassifier(lr = 1e-3,
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