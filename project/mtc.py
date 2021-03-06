# -*- coding: utf-8 -*-
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, precision_recall
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from pytorch_lightning.utilities.seed import seed_everything
from base_classifier import BaseClassifier


class MTCClassifier(BaseClassifier):

    def __init__(self, hparams, tokenizer, collator, encoder_model,
                 batch_size, nr_frozen_epochs, encoder_learning_rate, learning_rate,):
        super().__init__(hparams, tokenizer, collator, encoder_model,
                         batch_size, nr_frozen_epochs,
                         #  label_encoder,
                         encoder_learning_rate, learning_rate)

        self._num_classes = 5

        # build model
        self._build_model(self.encoder_model)

        # Loss criterion initialization.
        self._build_loss()

        if nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = nr_frozen_epochs

    def num_classes(self):
        return self._num_classes

    def encoder(self):
        return self._encoder

    def classification_head(self):
        return self._classification_head

    def _build_loss(self):
        self._loss_fn = nn.CrossEntropyLoss()

    def _get_metrics(self, logits, labels):
        preds = torch.argmax(logits, dim=1)

        # acc
        acc = (torch.sum(labels == preds).type_as(labels) 
               / (len(labels) * 1.0))

        # f1
        f1_ = f1(preds, labels, num_classes=self.num_classes(), average='macro')

        # precision and recall
        precision_, recall_ = precision_recall(
            preds, labels, num_classes=self.num_classes(), average='macro')

        return acc, f1_, precision_, recall_

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        return self._loss_fn(predictions["logits"], targets["labels"])

    def predict(self, sample: dict) -> dict:
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.collator(
                [sample], prepare_target=False)
            model_out = self.forward(model_input)
            logits = model_out["logits"].numpy()

            sample["predicted_label"] = np.argmax(logits, axis=1)[0]

        return sample


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
