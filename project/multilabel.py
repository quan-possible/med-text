# -*- coding: utf-8 -*-
from base_classifier import BaseClassifier
from label_attention import LabelAttentionLayer
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from utils import F1WithLogitsLoss, mask_fill

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from argparse import Namespace
from math import floor, ceil

import pytorch_lightning as pl
from transformers import AutoModel
from torchnlp.utils import lengths_to_mask
from torchmetrics.functional import accuracy, f1, precision_recall
from pytorch_lightning.utilities.seed import seed_everything


class MultiLabelClassifier(BaseClassifier):

    def __init__(self, desc_tokens, tokenizer, collator, num_classes, train_size, hparams, *args, **kwargs):
        super().__init__(desc_tokens, tokenizer, collator, num_classes, train_size, hparams, *args, **kwargs)
        
    def _build_loss(self):
        # self._loss_fn = nn.BCEWithLogitsLoss(
        #     # pos_weight=torch.tensor([5, 15, 15, 15, 7, 5, 12, 4, 3, 7])
        # )
        self._loss_fn = F1WithLogitsLoss()

    def _get_metrics(self, logits, labels):
        normed_logits = torch.sigmoid(logits)
        # print(preds)

        # acc
        acc = accuracy(normed_logits, labels)

        # f1
        f1_ = f1(normed_logits, labels, num_classes=10, average=self.hparams.metric_averaging)

        # precision and recall
        precision_, recall_ = precision_recall(
            normed_logits, labels, num_classes=10, average=self.hparams.metric_averaging)

        return acc, f1_, precision_, recall_

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x num_classes] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss_fn(predictions["logits"], targets["labels"].type_as(predictions["logits"]))
    
    def predict(self, sample: str) -> dict:
        if self.training:
            self.eval()
        print(sample)
        with torch.no_grad():
            sample_dict = {"text": sample}
            model_input, _ = self.collator(sample_dict, prepare_targets=False)
            model_out = self.forward(model_input)
            logits = model_out["logits"].numpy()
            output = np.argmax(logits, axis=1)
            

        return output


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
        hparams.dataset, hparams.batch_size,
        hparams.num_workers,
    )

    model = HOCClassifier(
        hparams, tokenizer, collator, hparams.encoder_model,
        hparams.batch_size, hparams.num_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
