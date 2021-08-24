# -*- coding: utf-8 -*-
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from base_classifier import BaseClassifier

from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, precision_recall
from pytorch_lightning.utilities.seed import seed_everything


class MultiClassClassifier(BaseClassifier):

    def __init__(self, desc_tokens, tokenizer, collator, num_classes, train_size, hparams, *args, **kwargs):
        super().__init__(desc_tokens, tokenizer, collator, num_classes, train_size, hparams, *args, **kwargs)

    def _build_loss(self):
        self._loss_fn = nn.CrossEntropyLoss()

    def _get_metrics(self, logits, labels):
        preds = torch.argmax(logits, dim=1)

        # acc
        acc = (torch.sum(labels == preds).type_as(labels)
               / (len(labels) * 1.0))

        # f1
        f1_ = f1(preds, labels, num_classes=self.num_classes, average='macro')

        # precision and recall
        precision_, recall_ = precision_recall(
            preds, labels, num_classes=self.num_classes, average='macro')

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
        return self._loss_fn(predictions["logits"], targets["labels"])

    def predict(self, sample: str):
        if self.training:
            self.eval()

        with torch.no_grad():
            sample_dict = {"text": sample}
            model_input, _ = self.collator(sample_dict, prepare_targets=False)
            model_out = self.forward(model_input)
            logits = model_out["logits"]
            # print("logit shape:")
            # print(logits.shape)

            output = np.argmax(logits, axis=1)

        return output


if __name__ == "__main__":

    seed_everything(69)

    hparams = Namespace(
        encoder_model="bert-base-cased",
        data_path="./project/data",
        dataset="mtc",
        batch_size=2,
        num_workers=12,
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
        hparams.dataset, hparams.batch_size,
        hparams.num_workers,
    )

    model = MTCClassifier(
        hparams, tokenizer, collator, hparams.encoder_model,
        hparams.batch_size, hparams.nr_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
