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
from utils import F1WithLogitsLoss

class HOCClassifier(BaseClassifier):

    def __init__(self, hparams, tokenizer, collator, encoder_model,
                 batch_size, nr_frozen_epochs, encoder_learning_rate, learning_rate,):
        super().__init__(hparams, tokenizer, collator, encoder_model,
                         batch_size, nr_frozen_epochs,
                         #  label_encoder,
                         encoder_learning_rate, learning_rate)
        
        self._num_classes = 10 
        
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

    # def _f1_loss(output, target):


    def _build_loss(self):
        self._loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([5, 15, 15, 15, 7, 5, 12, 4, 3, 7])
        )
        # self._loss_fn = F1WithLogitsLoss()
        
    def _get_metrics(self, logits, labels):
        shrinked_logits = torch.sigmoid(logits)
        # print(preds)

        # acc
        acc = accuracy(shrinked_logits, labels)

        # f1
        f1_ = f1(shrinked_logits, labels, num_classes=10, average=self.hparams.metric_averaging)

        # precision and recall
        precision_, recall_ = precision_recall(shrinked_logits, labels, num_classes=10, average=self.hparams.metric_averaging)
        
        return acc, f1_, precision_, recall_
        
    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        return self._loss_fn(predictions["logits"], targets["labels"].float())
    
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
        hparams.dataset, hparams.batch_size, 
        hparams.num_workers,
    )

    model = HOCClassifier(
        hparams, tokenizer, collator, hparams.encoder_model,
        hparams.batch_size, hparams.nr_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
