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

from Extras.loadconfigs import (S_SIAMESE_PATH,
                                S_NUM_CLASSES,
                                S_LABELS_NAME,
                                DEPTH
                                )
from torchmetrics.functional import confusion_matrix
from sklearn.metrics import classification_report
import torchmetrics as metrics
import seaborn as sns

class LTNClassifier(pl.LightningModule):
    def __init__(self, 
                 streamtype = 'img',
                 lr = 1e-3,
                 truncate_train = None,
                 truncate_val = None
                 ):
        super(LTNClassifier, self).__init__()
        assert streamtype in ['flo', 'img'], "streamtype must be either 'flo' or 'img'"
        self.streamtype = streamtype
        self.model = Classifier(LTNSiamese.load_from_checkpoint(S_SIAMESE_PATH).model,
                                num_classes=S_NUM_CLASSES)
        self.lr = lr
        self.truncate_train = truncate_train
        self.truncate_val = truncate_val
        
        self.auroc = metrics.AUROC(num_classes = S_NUM_CLASSES)
        
        self.train_accuracy_K1 = metrics.Accuracy(num_classes = S_NUM_CLASSES)
        self.train_accuracy_K3 = metrics.Accuracy(num_classes = S_NUM_CLASSES, top_k=3)
        self.train_accuracy_K5 = metrics.Accuracy(num_classes = S_NUM_CLASSES, top_k=5)
        
        self.val_accuracy_K1 = metrics.Accuracy(num_classes = S_NUM_CLASSES)
        self.val_accuracy_K3 = metrics.Accuracy(num_classes = S_NUM_CLASSES, top_k=3)
        self.val_accuracy_K5 = metrics.Accuracy(num_classes = S_NUM_CLASSES, top_k=5)
        
        self.best_accuracy = 0
    
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
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
        y = batch['s_label']
        rots_1 = batch["rots_1"]
        prob, _ = self.model(x, softmax=True, rots_1 = rots_1, mode = 'img')
        loss = torch.nn.CrossEntropyLoss()(prob, y)
        
        self.log("Loss/train_loss", loss, prog_bar = True, on_step=True, on_epoch=True, sync_dist=True)
        return {'loss':loss,
                'preds':prob,
                'target':y,
                'log': {'Loss/train': loss,},
                }
    
    def training_step_end(self, outputs):
        self.train_accuracy_K1(outputs['preds'], outputs['target'])
        self.train_accuracy_K3(outputs['preds'], outputs['target'])
        self.train_accuracy_K5(outputs['preds'], outputs['target'])
        
        self.log("Accuracy/train_acc_K1", self.train_accuracy_K1, on_step=True, on_epoch=False, prog_bar = True)
        self.log("Accuracy/train_acc_K3", self.train_accuracy_K3, on_step=True, on_epoch=False, prog_bar = True)
        self.log("Accuracy/train_acc_K5", self.train_accuracy_K5, on_step=True, on_epoch=False, prog_bar = True)
    
    def validation_step(self, batch, batch_idx):
        stopExec()
        x = batch[self.streamtype]
        y = batch['s_label']
        rots_1 = batch["rots_1"]
        
        prob, _ = self.model(x, softmax=True, rots_1 = rots_1, mode = 'img')
        loss = torch.nn.CrossEntropyLoss()(prob, y)
        
        self.log("Loss/val_loss", loss, prog_bar = True, on_step=True, on_epoch=True, sync_dist=True)
        
        return {'val_loss':loss,
                'preds':prob,
                'target':y,
                'log': {'Loss/val': loss,},
                }
    def validation_step_end(self, outputs):
        self.val_accuracy_K1(outputs['preds'], outputs['target'])
        self.val_accuracy_K3(outputs['preds'], outputs['target'])
        self.val_accuracy_K5(outputs['preds'], outputs['target'])
        
        self.log("Accuracy/val_acc_K1", self.val_accuracy_K1, on_step=True, on_epoch=False, prog_bar = True, sync_dist = True)
        self.log("Accuracy/val_acc_K3", self.val_accuracy_K3, on_step=True, on_epoch=False, prog_bar = True, sync_dist = True)
        self.log("Accuracy/val_acc_K5", self.val_accuracy_K5, on_step=True, on_epoch=False, prog_bar = True, sync_dist = True)
    
    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        
        preds = sync_tensor_across_gpus(preds)
        targets = sync_tensor_across_gpus(targets)
        
        
        self.auroc.update(preds, targets.long())
        self.log('auroc', self.auroc, prog_bar=True, on_epoch=True, sync_dist = True)
        
        current_accuracy = self.val_accuracy_K1.compute()
        
        try:
            _, y_pred = torch.max(preds, 1)
            y_pred = y_pred.detach().cpu().numpy().reshape(-1)
            y_true = targets.detach().cpu().numpy().reshape(-1)
            
            report = classification_report(y_true, y_pred, target_names=S_LABELS_NAME)
            if current_accuracy>self.best_accuracy:    
                with open("reports/SC_SIAMESE_CONFUSION_MATRIX_PR_Report.txt","w") as f:
                    f.write(report)
            
            self.logger.experiment.add_text("SC_PR_Report", report, self.current_epoch)
        except:
            pass
        
        cmatrix = confusion_matrix(preds, targets, num_classes=S_NUM_CLASSES)
        df_cm = pd.DataFrame(cmatrix.detach().cpu().numpy(), index = S_LABELS_NAME, columns=S_LABELS_NAME)
        if current_accuracy>self.best_accuracy:    
            df_cm.to_csv(f'reports/SC_SIAMESE_CONFUSION_MATRIX.csv')
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("SC_SIAMESE_CONFUSION_MATRIX", fig_, self.current_epoch)
        
        if current_accuracy>self.best_accuracy:
            self.best_accuracy = current_accuracy
    
    def train_dataloader(self):
        loader = getEgoLoader(mode = 'train', 
                              shuffle = True, 
                              siamese = False, 
                              focus = False, 
                              flo = False, 
                              img = True, 
                              low = -1, 
                              high = 1, 
                              double_rots = True,
                              truncate = self.truncate_train)
        return loader
        
    def val_dataloader(self):
        loader = getEgoLoader(mode = 'test', 
                              shuffle = True, 
                              siamese = False, 
                              focus = False, 
                              flo = False, 
                              img = True, 
                              low = -1, 
                              high = 1, 
                              double_rots = True,
                              truncate = self.truncate_val)
        return loader

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="cache",
    filename="SPATIAL_SIAMESE_CLASSIFIER",
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
    
    model = LTNClassifier(siamese_path = f'cache/DEPTH_{DEPTH}_MOTION_SIAMESE.ckpt',
                          num_classes = 10
                          )
    trainer.fit(model)