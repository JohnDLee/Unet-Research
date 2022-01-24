import torch
import sys
import os
import pytorch_lightning as pl

from utils.utils_unet import UNet

class BaseUNetTraining(pl.LightningModule):
    """ base training module for UNet, alter predict step for more predictions"""

    def __init__(self, model, loss_fcn, optimizer):
        super().__init__()
        self._model = model
        self._loss_fcn = loss_fcn
        self._optimizer = optimizer

    
    def forward(self, x):
        return self._model(x)
    
    def training_step(self, batch, batch_idx):
        im_batch, gt, mask = batch
        im_batch.requires_grad=True

        segmentation = self._model(im_batch)

        # mask
        segmentation = segmentation * mask
        gt = gt * mask
        
        loss = self._loss_fcn(segmentation, gt)
        # recalculate loss based on mask
        loss *= (segmentation.numel() / mask.count_nonzero())
        
        # log every 10
        if batch_idx % 10:
            self.log('train_loss', loss, prog_bar=True, logger=True,  on_step = True, on_epoch=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        im_batch, gt, mask = batch

        segmentation = self._model(im_batch)

        # mask
        segmentation = segmentation * mask
        gt = gt * mask
        
        loss = self._loss_fcn(segmentation, gt)
        # recalculate loss based on mask
        loss *= (segmentation.numel() / mask.count_nonzero())
        
        # log each one
        self.log('val_loss', loss, prog_bar=True, logger=True,  on_step = True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self._optimizer

    def test_step(self, batch, batch_idx):
        im_batch, _,  mask = batch

        segmentation = self._model(im_batch)

        # mask
        segmentation = segmentation * mask
 
        return segmentation
    
    def predict_step(self, batch, batch_idx):
        im_batch, gt, mask = batch
        segmentation = self._model(im_batch)

        segmentation = segmentation * mask
        
        return batch_idx, segmentation, im_batch, gt
    
