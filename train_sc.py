
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from Extras.loadconfigs import DEPTH
from Classifier.sc import *

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="cache",
    filename=f"DEPTH_{DEPTH}_SPATIAL_CLASSIFIER",
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