from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from Utils.dataloader import *
from Utils.utils import *
from Classifier.cat import *


if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="cache",
    filename="CAT_CLASSIFIER",
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
    
    model = CatClassifier(lr = 1e-3,
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