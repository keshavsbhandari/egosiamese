import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics.functional import confusion_matrix
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Model.model import *
from Utils.egoloader import *
from Utils.utils import *
from Siamese.m_siamese import *
import torchmetrics as metrics
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from Extras.loadconfigs import (M_SIAMESE_PATH,
                                M_NUM_CLASSES,
                                M_LABELS_NAME,
                                SERVER

                                )

class LTNClassifier(pl.LightningModule):
    def __init__(self,
                 streamtype = 'flo',
                 lr = 0.000036983,
                 truncate_train = None,
                 truncate_val = None
                 ):
        super(LTNClassifier, self).__init__()
        assert streamtype in ['flo', 'img'], "streamtype must be either 'flo' or 'img'"
        self.streamtype = streamtype
        self.model = Classifier(LTNSiamese(streamtype=streamtype).load_from_checkpoint(M_SIAMESE_PATH).model,
                                num_classes=M_NUM_CLASSES)
        self.lr = lr
        self.truncate_train = truncate_train
        self.truncate_val = truncate_val
        
        self.train_accuracy_K1 = metrics.Accuracy(num_classes = M_NUM_CLASSES)
        self.train_accuracy_K3 = metrics.Accuracy(num_classes = M_NUM_CLASSES, top_k=3)
        self.train_accuracy_K5 = metrics.Accuracy(num_classes = M_NUM_CLASSES, top_k=5)
        
        self.val_accuracy_K1 = metrics.Accuracy(num_classes = M_NUM_CLASSES)
        self.val_accuracy_K3 = metrics.Accuracy(num_classes = M_NUM_CLASSES, top_k=3)
        self.val_accuracy_K5 = metrics.Accuracy(num_classes = M_NUM_CLASSES, top_k=5)
        
        self.best_accuracy = 0

        self.auroc = metrics.AUROC(num_classes = M_NUM_CLASSES)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.5, 0.599), eps=1e-08, weight_decay=0.001, amsgrad=False)
        return {
            "optimizer":optimizer,
            "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95),
            "interval":"epoch",
            "monitor":"Loss/val_loss",
            "frequency":1,
        }
        
    def training_step(self, batch, batch_idx):
        stopExec()
        x = batch[self.streamtype]
        y = batch['m_label']
        rots_1 = batch["rots_1"]
        
        prob, _ = self.model(x, softmax=True, rots_1 = rots_1, mode = 'flo')
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
        
        self.log("Accuracy/train_acc_K1", self.train_accuracy_K1, on_step=False, on_epoch=True, prog_bar = True)
        self.log("Accuracy/train_acc_K3", self.train_accuracy_K3, on_step=False, on_epoch=True, prog_bar = True)
        self.log("Accuracy/train_acc_K5", self.train_accuracy_K5, on_step=False, on_epoch=True, prog_bar = True)
        
    
    def validation_step(self, batch, batch_idx):
        stopExec()
        x = batch[self.streamtype]
        y = batch['m_label']
        rots_1 = batch["rots_1"]
        
        prob, _ = self.model(x, softmax=True, rots_1 = rots_1, mode = 'flo')
        loss = torch.nn.CrossEntropyLoss()(prob, y)
        
        self.log("Loss/val_loss_epoch", loss, prog_bar = True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'val_loss':loss,
                'preds':prob,
                'target':y,
                'log': {'Loss/val': loss,},
                }
    
    def validation_step_end(self, outputs):
        self.val_accuracy_K1(outputs['preds'], outputs['target'])
        self.val_accuracy_K3(outputs['preds'], outputs['target'])
        self.val_accuracy_K5(outputs['preds'], outputs['target'])
        
        self.log("Accuracy/val_acc_K1", self.val_accuracy_K1, on_step=False, on_epoch=True, prog_bar = True)
        self.log("Accuracy/val_acc_K3", self.val_accuracy_K3, on_step=False, on_epoch=True, prog_bar = True)
        self.log("Accuracy/val_acc_K5", self.val_accuracy_K5, on_step=False, on_epoch=True, prog_bar = True)
    
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
            
            report = classification_report(y_true, y_pred, target_names=M_LABELS_NAME)
            if current_accuracy>self.best_accuracy:    
                with open(f"reports/{SERVER}_MC_SIAMESE_CONFUSION_MATRIX_PR_Report.txt","w") as f:
                    f.write(report)
            
            self.logger.experiment.add_text(f"{SERVER}_MC_PR_Report", report, self.current_epoch)
        except:
            pass
        
        cmatrix = confusion_matrix(preds, targets, num_classes=M_NUM_CLASSES)
        df_cm = pd.DataFrame(cmatrix.detach().cpu().numpy(), index = M_LABELS_NAME, columns=M_LABELS_NAME)
        if current_accuracy>self.best_accuracy:    
            df_cm.to_csv(f'reports/{SERVER}_MC_SIAMESE_CONFUSION_MATRIX.csv')
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure(f"{SERVER}_MC_SIAMESE_CONFUSION_MATRIX", fig_, self.current_epoch)
        
        if current_accuracy>self.best_accuracy:
            self.best_accuracy = current_accuracy
    
    def find_lr(self, init_value, final_value):
        train_loader = self.train_dataloader()
        number_in_epoch = len(train_loader) - 1
        update_step = (final_value / init_value) ** (1 / number_in_epoch)
        lr = init_value
        optimizer = self.configure_optimizers().get('optimizer')
        optimizer.param_groups[0]["lr"] = lr
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        iterator = tqdm(train_loader, desc="Current lr=XX.XX Steps=XX Loss=XX.XX Best lr=XX.XX ")
        for i, batch in enumerate(iterator):
            batch_num += 1
            x = batch[self.streamtype]
            y = batch['m_label']
            rots_1 = batch["rots_1"]
            
            optimizer.zero_grad()

            prob, _ = self.model(x, softmax=True, rots_1 = rots_1, mode = 'flo')
            loss = torch.nn.CrossEntropyLoss()(prob, y)

            # Crash out if loss explodes
            if batch_num > 1 and loss > 4 * best_loss:
                return log_lrs[10:-5], losses[10:-5]
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

        self.lr = best_lr
        plt.plot(logs, losses)
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.savefig(f"{self.streamtype} - Motion Classifier - Optimal lr curve.png")
        print("plot saved")
            
    def train_dataloader(self):
        loader = getEgoLoader(mode = 'train', 
                              shuffle = True, 
                              siamese = False, 
                              focus = False, 
                              flo = True, 
                              img = False, 
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
                              flo = True, 
                              img = False, 
                              low = -1, 
                              high = 1, 
                              double_rots = True,
                              truncate = self.truncate_val)
        return loader

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="cache",
    filename="MOTION_SIAMESE_CLASSIFIER",
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
    
    model = LTNClassifier()
    trainer.fit(model)