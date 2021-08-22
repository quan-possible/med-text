# -*- coding: utf-8 -*-
import logging as log
from math import ceil
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from transformers import AutoModel
from abc import abstractmethod

import pytorch_lightning as pl
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from torchnlp.utils import lengths_to_mask
from pytorch_lightning.utilities.seed import seed_everything
from utils import mask_fill, get_lr_schedule, str2bool


class BaseClassifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, desc_tokens, tokenizer, collator, hparams, * args, **kwargs
                 #  encoder_model,
                 #  batch_size, num_frozen_epochs,
                 #  encoder_learning_rate, learning_rate,
                 ) -> None:
        super(BaseClassifier, self).__init__()

        # self.hparams = hparams
        self.desc_tokens = desc_tokens
        self.tokenizer = tokenizer
        self.collator = collator
        self.save_hyperparameters(hparams)

    @property
    @abstractmethod
    def num_classes(self):
        pass


    @abstractmethod
    def _get_metrics(self, logits, labels):
        pass

    @abstractmethod
    def predict(self, sample: dict):
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        pass

    @abstractmethod
    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x num_classes] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        pass

    @abstractmethod
    def _build_model(self) -> None:
        pass

    @abstractmethod
    def forward(self, tokens_lengths):
        """ Usual pytorch forward function. 
        :param tokens_lengths: tuple of:
            - text sequences [batch_size x src_seq_len]
            - lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        pass

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._frozen = True

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        encoder_names = ['encoder']
        label_attn = ['label_attn']
        param_groups = [
            {
                'params': [p for n, p in self.named_parameters()
                           if any(nd in n for nd in encoder_names)],
                'name': "encoder",
                'lr': self.hparams.encoder_learning_rate,
                'weight_decay': self.hparams.weight_decay_encoder,
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if any(nd in n for nd in label_attn)],
                'name': "label_attention",
                'lr': self.hparams.label_attn_lr,
                'weight_decay': self.hparams.weight_decay_encoder,
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in (encoder_names + label_attn))],
                'name': 'non-encoder',
                'weight_decay': self.hparams.weight_decay_nonencoder,
            },
        ]

        self.optimizer = optim.AdamW(param_groups, lr=self.hparams.learning_rate)

        steps_per_epoch = ceil(1303 / (self.hparams.batch_size * 2))
        self.lr_scheduler = get_lr_schedule(
            param_groups=param_groups, encoder_indices=[0], optimizer=self.optimizer,
            scheduler_epochs=self.hparams.scheduler_epochs, 
            num_frozen_epochs=self.hparams.num_frozen_epochs,
            steps_per_epoch=steps_per_epoch, 
            warmup_pct=[self.hparams.warmup_pct, self.hparams.warmup_pct],
            smallest_lr_pct=[self.hparams.smallest_lr_pct_encoder,
                             self.hparams.smallest_lr_pct_lbl_attn, 
                             self.hparams.smallest_lr_pct_nonencoder],
        )

        # self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=[5e-05, 1e-03], epochs=self.hparams.max_epochs,
        #     steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='linear',
        #     cycle_momentum=False, div_factor=2.50, final_div_factor=20.0,
        #     three_phase=False, last_epoch=-1, verbose=True
        # )

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'interval': 'step',
            }
        }

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.hparams.num_frozen_epochs:
            self.unfreeze_encoder()

    def training_step(self, batch: tuple, batch_idx) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_idx: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss = self.loss(model_out, targets)

        self.log("loss", loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        val_loss = self.loss(model_out, targets)

        labels = targets["labels"]   # (batch_size, num_labels)
        logits = model_out["logits"]    # (batch_size, num_labels)

        val_acc, val_f1, val_precision, val_recall = \
            self._get_metrics(logits, labels)

        loss_acc = OrderedDict({"val_loss": val_loss, "val_acc": val_acc})
        metrics = OrderedDict({"val_f1": val_f1,
                               "val_precision": val_precision,
                               "val_recall": val_recall})

        self.log_dict(loss_acc, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("hp_metric", val_f1)

        # # can also return just a scalar instead of a dict (return loss_val)
        return loss_acc

    def test_step(self, batch: tuple, batch_idx: int,) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)

        labels = targets["labels"]   # (batch_size, num_labels)
        logits = model_out["logits"]    # (batch_size, num_labels)

        test_acc, test_f1, test_precision, test_recall = \
            self._get_metrics(logits, labels)

        metrics = OrderedDict({"test_acc": test_acc,
                               "test_f1": test_f1,
                               "test_precision": test_precision,
                               "test_recall": test_recall})

        self.log_dict(metrics, sync_dist=True)

        # can also return just a scalar instead of a dict (return loss_val)
        return metrics

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) \
            -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """

        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--num_frozen_epochs",
            default=10,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )

        parser.add_argument(
            "--metric_averaging",
            default="micro",
            type=str,
            help="Averaging methods for validation metrics (micro, macro,...)",
        )
        
        parser.add_argument(
            "--num_heads",
            default=12,
            type=int,
            help="Averaging methods for validation metrics (micro, macro,...)",
        )

        parser.add_argument(
            "--warmup_pct",
            default=0.1,
            type=float,
            help="Percentage of training steps used for warmup",
        )

        parser.add_argument(
            "--scheduler_epochs",
            default=30,
            type=int,
            help="Number of epochs the scheduler for the encoder is activated",
        )
        
        parser.add_argument(
            "--weight_decay_encoder",
            default=0.05,
            type=float,
            help="Weight decay for encoder",
        )
        
        parser.add_argument(
            "--weight_decay_nonencoder",
            default=0.1,
            type=float,
            help="Weight decay for non-encoder",
        )
        
        parser.add_argument(
            "--smallest_lr_pct_encoder",
            default=0.01,
            type=float,
            help="Smallest encoder learning rate being a percentage of the default learning rate",
        )

        parser.add_argument(
            "--smallest_lr_pct_lbl_attn",
            default=0.001,
            type=float,
            help="Smallest non-encoder learning rate being a percentage of the default learning rate",
        )
        
        parser.add_argument(
            "--smallest_lr_pct_nonencoder",
            default=0.4,
            type=float,
            help="Smallest non-encoder learning rate being a percentage of the default learning rate",
        )
        
        parser.add_argument(
            "--n_lbl_attn_layer",
            default=1,
            type=int,
            help="Number of label attention layers.",
        )

        parser.add_argument(
            "--static_desc_emb", 
            type=str2bool, 
            default=False,
            help="Whether to update description embedding using BERT"
        )
        
        parser.add_argument(
            "--label_attn_lr",
            type=float,
            default=1e-04,
            help="Learning rate for label attention layers"
        )

        return parser

    " DEPRECATED "
    # def validation_epoch_end(self, outputs: list) -> dict:
    #     """ Function that takes as input a list of dictionaries returned by the validation_step
    #     function and measures the model performance accross the entire validation set.

    #     Returns:
    #         - Dictionary with metrics to be added to the lightning logger.
    #     """
    #     val_loss_mean = 0
    #     val_acc_mean = 0
    #     for output in outputs:
    #         val_loss = output["val_loss"]

    #         # reduce manually when using dp
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_loss = torch.mean(val_loss)
    #         val_loss_mean += val_loss

    #         # reduce manually when using dp
    #         val_acc = output["val_acc"]
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_acc = torch.mean(val_acc)

    #         val_acc_mean += val_acc

    #     val_loss_mean /= len(outputs)
    #     val_acc_mean /= len(outputs)
    #     tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
    #     result = {
    #         "progress_bar": tqdm_dict,
    #         "log": tqdm_dict,
    #         "val_loss": val_loss_mean,
    #         "val_acc": val_acc_mean,
    #     }


    #     return result
if __name__ == "__main__":

    seed_everything(69)

    hparams = Namespace(
        encoder_model="bert-base-cased",
        data_path="./project/data",
        dataset="hoc",
        batch_size=2,
        num_workers=2,
        random_sampling=False,
        num_frozen_epochs=1,
        encoder_learning_rate=1e-05,
        learning_rate=3e-05,
        tgt_txt_col="TEXT",
        tgt_lbl_col="LABEL",
    )

    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, hparams.num_workers,
        hparams.tgt_txt_col, hparams.tgt_lbl_col,
    )

    num_classes = datamodule.num_classes

    model = HOCClassifier(
        hparams, tokenizer, collator, hparams.encoder_model,
        hparams.batch_size, num_classes, hparams.num_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
