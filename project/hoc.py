# -*- coding: utf-8 -*-
from base_classifier import BaseClassifier

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


class HOCClassifier(BaseClassifier):

    def __init__(self, desc_tokens, tokenizer, collator, hparams, static_desc_emb=False, * args, **kwargs):
        super().__init__(desc_tokens, tokenizer, collator, hparams, *args, **kwargs)

        self._num_classes = 10

        # build model
        self._build_model()

        # Loss criterion initialization.
        self._build_loss()

        self.desc_tokens = desc_tokens  # (batch_size, seq_len)

        self.static_desc_emb = static_desc_emb
        if self.static_desc_emb:
            with torch.no_grad():
                self.desc_emb = self._process_tokens(self.desc_tokens)[:, 0, :].squeeze(dim=1)
                
        if self.hparams.num_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

    @property
    def num_classes(self):
        return self._num_classes


        
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
