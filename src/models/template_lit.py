# Copyright (c) 2023, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os
from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Recall
from torchmetrics.classification import F1Score
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.roc import ROC


class LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        exp_name=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.net = net
        self.n_classes = self.net.num_classes
        self.exp_name = exp_name

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.n_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=self.n_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.n_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=self.n_classes)

        # self.train_roc = ROC(task="multiclass", num_classes=self.n_classes)
        # self.val_roc = ROC(task="multiclass", num_classes=self.n_classes)
        # self.test_roc = ROC(task="multiclass", num_classes=self.n_classes)

        self.train_uar = Recall(
            task="multiclass", average="macro", num_classes=self.n_classes
        )
        self.val_uar = Recall(
            task="multiclass", average="macro", num_classes=self.n_classes
        )
        self.test_uar = Recall(
            task="multiclass", average="macro", num_classes=self.n_classes
        )

        self.train_auroc = AUROC(
            task="multiclass",
            num_classes=self.n_classes,
            average="macro",
            thresholds=None,
        )
        self.val_auroc = AUROC(
            task="multiclass",
            num_classes=self.n_classes,
            average="macro",
            thresholds=None,
        )
        self.test_auroc = AUROC(
            task="multiclass",
            num_classes=self.n_classes,
            average="macro",
            thresholds=None,
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_uar_best = MaxMetric()

        # Dict for saving values on test epoch end
        self.test_epoch_end_lst = []

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_uar.reset()
        self.val_auroc.reset()
        self.val_uar_best.reset()

    def model_step(self, batch: Any):
        x, y, vids = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, logits, vids

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_uar(preds, targets)
        # self.train_roc(logits, targets)
        self.train_auroc(logits, targets)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/uar",
            self.train_uar,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # self.log(
        #     "train/roc",
        #     self.train_roc,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        self.log(
            "train/auroc",
            self.train_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_uar(preds, targets)
        # self.val_roc(logits, targets)
        self.val_auroc(logits, targets)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/uar",
            self.val_uar,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # self.log(
        #     "val/roc",
        #     self.val_roc,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        self.log(
            "val/auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # get current val metric
        # acc = self.val_acc.compute()
        uar = self.val_uar.compute()
        # f1 = self.val_f1.compute()
        # auroc = self.val_auroc.compute()
        # roc = self.val_roc.compute()

        # update best so far val acc
        # self.val_acc_best(acc)
        self.val_uar_best(uar)
        # self.val_f1_best(f1)
        # self.val_auroc_best(auroc)
        # self.val_roc_best(roc)

        self.log(
            "val/uar_best",
            self.val_uar_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        # self.log(
        #     "val/f1_best",
        #     self.val_f1_best.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "val/auroc_best",
        #     self.val_auroc_best.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "val/roc_best",
        #     self.val_roc_best.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

    def test_step(self, batch: Any, batch_idx: int):
        x, y, _ = batch
        loss, preds, targets, logits, vids = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_uar(preds, targets)
        # self.test_roc(logits, targets)
        self.test_auroc(logits, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/uar",
            self.test_uar,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # self.log(
        #     "test/roc",
        #     self.test_roc,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        self.log(
            "test/auroc",
            self.test_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Save
        self.test_epoch_end_lst.append(
            {
                "vids": vids,
                "logits": logits,
                "labels": targets,
                "loss": loss,
                "preds": preds,
            }
        )

    def on_test_epoch_end(self):
        # Aggregate logits and labels from all test steps
        all_logits = torch.cat([x["logits"] for x in self.test_epoch_end_lst], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.test_epoch_end_lst], dim=0)
        all_preds = torch.cat([x["preds"] for x in self.test_epoch_end_lst], dim=0)
        all_inputs = torch.cat([x["vids"] for x in self.test_epoch_end_lst], dim=0)
        save_dict = {
            "logits": all_logits,
            "labels": all_labels,
            "preds": all_preds,
            "inputs": all_inputs,
        }

        # Save
        rootdir = os.environ["PROJECT_ROOT"]
        savedir = f"{rootdir}/pkl/results/{self.exp_name}.pth"
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
        torch.save(save_dict, savedir)

        # Free up memory
        self.test_epoch_end_lst.clear()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitModule(None, None, None)
