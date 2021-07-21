# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import pandas as pd

from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from abc import abstractmethod
from transformers import get_linear_schedule_with_warmup

import pytorch_lightning as pl
from torchnlp.utils import lengths_to_mask
from pytorch_lightning.utilities.seed import seed_everything
from utils import mask_fill


class BaseClassifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams, desc_tokens, tokenizer, collator,
                 encoder_model, batch_size, nr_frozen_epochs,
                 encoder_learning_rate, learning_rate,
                 num_heads, num_warmup_steps, num_training_steps,
                 metric_averaging,
                 ) -> None:
        super(BaseClassifier, self).__init__()

        # self.hparams = hparams
        self.desc_tokens = desc_tokens
        self.tokenizer = tokenizer
        self.collator = collator
        self.nr_frozen_epochs = nr_frozen_epochs
        self.batch_size = batch_size
        self.encoder_model = encoder_model
        self.num_heads = num_heads
        self.encoder_learning_rate = encoder_learning_rate
        self.learning_rate = learning_rate
        self.num_metrics = 3
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.metric_averaging = metric_averaging

        self.save_hyperparameters(hparams)

    @property
    @abstractmethod
    def num_classes(self):
        pass
    
    # @property
    # @abstractmethod
    # def desc_emb(self):
    #     pass

    @property
    @abstractmethod
    def encoder(self):
        pass
    
    @property
    def label_attn(self):
        pass

    @property
    @abstractmethod
    def classification_head(self):
        pass

    @abstractmethod
    def get_metrics(self, logits, labels):
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
    def forward(self, tokens_dict):
        pass
    
    @abstractmethod
    def _build_model(self, encoder_model) -> None:
        pass

    def _process_tokens(self, tokens_dict, type_as_tensor=None):
        tokens, lengths = tokens_dict['tokens'], \
            tokens_dict['lengths']
        tokens = tokens[:, : lengths.max()]

        if type_as_tensor != None:
            tokens = tokens.to(type_as_tensor.device).detach()
            
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch. In other words, use this when using DataParallel.
        mask = lengths_to_mask(lengths, device=tokens.device)
        
        # Run BERT model.
        # output is (batch_size, sequence_length, hidden_size)
        emb = self.encoder(tokens, mask).last_hidden_state

        return emb

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
        
        # # Optimizer:
        # no_decay = ['_classification_head', '_label_attn']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.01},
        #     {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        # parameters = [
        #     {"params": self.classification_head.parameters()}, 
        #     {"params": self.label_attn.parameters()},
        #     {"params": self.encoder.parameters(), 
        #      "lr": self.encoder_learning_rate}
        # ]
        
        # self.optimizer = optim.Adam(parameters,
        #                                 lr=self.learning_rate)
        
        # scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, self.num_warmup_steps,
        #     self.num_training_steps)
        # return [self.optimizer], [scheduler]
        
        
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.learning_rate)
        
        return [self.optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
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

        return loss

    def validation_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        val_loss = self.loss(model_out, targets)

        labels = targets["labels"]   # (batch_size, num_labels)
        logits = model_out["logits"]    # (batch_size, num_labels)

        val_acc, val_f1, val_precision, val_recall, p_class_score = \
            self.get_metrics(logits, labels)

        loss_acc = OrderedDict({"val_loss": val_loss, "val_acc": val_acc})
        metrics = OrderedDict({"val_f1": val_f1,
                               "val_precision": val_precision,
                               "val_recall": val_recall})

        self.log_dict(loss_acc, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("hp_metric", val_f1)

        # for param_group in self.optimizer.param_groups:
        #     print(f"{param_group['lr']}")

        # # can also return just a scalar instead of a dict (return loss_val)
        return p_class_score
    
    # def validation_epoch_end(self, outputs) -> None:
    #     lbl_order = [5,9,8,3,1,6,4,2,0,7]
        
    #     len_outputs = len(outputs) 
    #     res = torch.zeros(self.num_metrics, self.num_classes)
    #     for output in outputs:
    #         res += output
            
    #     res /= len_outputs
    #     res = res[:,lbl_order]
    #     # self.p_class_metrics = res.type_as(outputs[0])
            
    #     print("Per class metrics: ")
    #     print("f1: ", res[0])
    #     print("precision: ", res[1])
    #     print("recall: ", res[2])
        
        # self.log("per_class_metrics", self.p_class_metrics, sync_dist=True)
        
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
            self.get_metrics(logits, labels)

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
            default=5e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=1e-03,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=15,
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
            default=8,
            type=int,
            help="Number of heads for label attention.",
        )
        
        parser.add_argument(
            "--num_warmup_steps",
            default=50,
            type=int,
            help="Number of learning rate warm up steps",
        )
        
        parser.add_argument(
            "--num_training_steps",
            default=500,
            type=int,
            help="Number of training steps before learning rate get to 0.",
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
        nr_frozen_epochs=1,
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
        hparams.batch_size, num_classes, hparams.nr_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
