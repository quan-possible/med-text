# -*- coding: utf-8 -*-
from base_classifier import BaseClassifier
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from utils import F1WithLogitsLoss, mask_fill


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from math import floor, ceil

import pytorch_lightning as pl
from transformers import AutoModel
from torchnlp.utils import lengths_to_mask
from torchmetrics.functional import accuracy, f1, precision_recall
from pytorch_lightning.utilities.seed import seed_everything


class HOCClassifier(BaseClassifier):

    def __init__(self, desc_tokens, tokenizer, collator, hparams, *args, **kwargs):
        super().__init__(desc_tokens, tokenizer, collator, hparams, *args, **kwargs)

        self._num_classes = 10

        # build model
        self._build_model()

        # Loss criterion initialization.
        self._build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def encoder(self):
        return self._encoder

    @property
    def classification_head(self):
        return self._classification_head

    def _build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        # pass

        self._encoder = AutoModel.from_pretrained(
            self.hparams.encoder_model, output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        if self.hparams.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            encoder_features = 128
        else:
            encoder_features = 768
            
        filter_sizes = [3, 5, 7]  # Only choose odd numbers
        self.dim_conv = int(768 / len(filter_sizes))

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=encoder_features,
                      out_channels=self.dim_conv,
                      kernel_size=size,
                      padding=floor(size / 2))
            for size in filter_sizes
        ])
        self.bn1d_list = nn.ModuleList([nn.BatchNorm1d(self.dim_conv) for _ in filter_sizes])

        # Classification head
        self._classification_head = nn.Sequential(
            nn.Linear(encoder_features, encoder_features * 2),
            nn.Tanh(),
            nn.Linear(encoder_features * 2, encoder_features),
            nn.Tanh(),
            nn.Linear(encoder_features, self.num_classes),
        )

    def _build_loss(self):
        self._loss_fn = nn.BCEWithLogitsLoss(
            # pos_weight=torch.tensor([5, 15, 15, 15, 7, 5, 12, 4, 3, 7])
        )
        # self._loss_fn = F1WithLogitsLoss()

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
    
    def forward(self, tokens_dict):
        """ Usual pytorch forward function. 
        :param tokens_dict: tuple of:
            - text sequences [batch_size x src_seq_len]
            - lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        # _process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        x = self._process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)

        x = x.transpose(2, 1)  # (batch_size, hidden_dim, seq_len)

        x_list = [F.relu(bn1d(conv1d(x))) for conv1d,
                  bn1d in zip(self.conv1d_list, self.bn1d_list)]

        x_list = [F.max_pool1d(x, kernel_size=x.size(2)) for x in x_list]

        x = torch.cat([x.squeeze(dim=2) for x in x_list], dim=1)

        logits = self.classification_head(x).squeeze()
        
        # tokens, lengths = tokens_dict['tokens'], \
        #     tokens_dict['lengths']
        # tokens = tokens[:, : lengths.max()]

        # # When using just one GPU this should not change behavior
        # # but when splitting batches across GPU the tokens have padding
        # # from the entire original batch
        # mask = lengths_to_mask(lengths, device=tokens.device)

        # # Run BERT model. output is (batch_size, sequence_length, hidden_size)
        # word_embeddings = self.encoder(tokens, mask).last_hidden_state

        # # Average Pooling
        # word_embeddings = mask_fill(
        #     0.0, tokens, word_embeddings, self.tokenizer.padding_index
        # )
        # sentemb = torch.sum(word_embeddings, 1)
        # sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()
        #                                      ).float().sum(1)
        # sentemb = sentemb / sum_mask

        # Classification head
        # logits = self.classification_head(sentemb)

        return {"logits": logits}

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
