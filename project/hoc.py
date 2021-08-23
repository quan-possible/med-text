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

    def _build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        # pass

        self.encoder = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            # hidden_dropout_prob=0.1,
            output_hidden_states=True,
        )

        # set the number of features our encoder model will return...
        if self.hparams.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768

        #-------------------------
        # DESCRIPTION EMBEDDINGS
        #-------------------------

        # # set the number of features our encoder model will return...
        # self._label_attn = nn.MultiheadAttention(
        #     self.encoder_features, self.hparams.num_heads, dropout=0.2,
        # )

        # # Classification head
        # self._classification_head = nn.Sequential(
        #     nn.Linear(self.encoder_features, self.encoder_features * 2),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(self.encoder_features * 2, self.encoder_features),
        #     nn.Tanh(),
        #     nn.Linear(self.encoder_features, 1),
        # )
        #----------------------------
        # NO HEAD
        #----------------------------

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.num_classes),
        )

        #----------------------------
        # CONVOLUTIONAL HEAD
        #----------------------------

        # filter_sizes = [3, 5, 7]  # Only choose odd numbers
        # self.dim_conv = int(768 / len(filter_sizes))
        #
        # # Conv Network
        # self.conv1d_list = nn.ModuleList([
        #     nn.Conv1d(in_channels=encoder_features,
        #               out_channels=self.dim_conv,
        #               kernel_size=size,
        #               padding=floor(size / 2))
        #     for size in filter_sizes
        # ])
        # self.bn1d_list = nn.ModuleList([nn.BatchNorm1d(self.dim_conv) for _ in filter_sizes])

        # # Classification head
        # self._classification_head = nn.Sequential(
        #     nn.Linear(self.encoder_features, self.encoder_features * 2),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(self.encoder_features * 2, self.encoder_features),
        #     nn.Tanh(),
        #     nn.Linear(self.encoder_features, self.num_classes),
        # )

        #---------------------------------
        # STACKED DESCRIPTION EMBEDDINGS
        #---------------------------------

        # label_attn_layer = LabelAttentionLayer(self.encoder_features, self.hparams.num_heads)
        # self.label_attn = self._get_clones(label_attn_layer, self.hparams.n_lbl_attn_layer)

        # self.classification_head = nn.Sequential(
        #     nn.Linear(self.encoder_features, self.encoder_features * 2),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(self.encoder_features * 2, self.encoder_features),
        #     nn.Tanh(),
        # )
        # self.final_fc = nn.Linear(self.encoder_features, self.num_classes)

        #---------------------------------
        # TRANSFORMERS
        #---------------------------------

        # trans_block = nn.Transformer(
        #     d_model=self.encoder_features,
        #     num_encoder_layers=2,
        #     num_decoder_layers=2,
        #     dropout=0.2,
        #     batch_first=True
        # )
        # self._label_attn = self._get_clones(trans_block, self.hparams.n_lbl_attn_layer)

        # self._classification_head = nn.Sequential(
        #     nn.Linear(self.encoder_features, self.encoder_features * 2),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(self.encoder_features * 2, self.encoder_features),
        #     nn.Tanh(),
        # )

        # self.final_fc = nn.Linear(self.encoder_features,self.num_classes)

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

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, tokens_dict):
        """ Usual pytorch forward function. 
        :param tokens_dict: tuple of:
            - text sequences [batch_size x src_seq_len]
            - lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        #----------------------------
        # CONVOLUTIONAL HEAD
        #----------------------------

        # # _process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        # x = self._process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)

        # x = x.transpose(2, 1)  # (batch_size, hidden_dim, seq_len)

        # x_list = [F.relu(bn1d(conv1d(x))) for conv1d,
        #           bn1d in zip(self.conv1d_list, self.bn1d_list)]

        # x_list = [F.max_pool1d(x, kernel_size=x.size(2)) for x in x_list]

        # x = torch.cat([x.squeeze(dim=2) for x in x_list], dim=1)

        # logits = self.classification_head(x)

        #----------------------------
        # NO HEAD
        #----------------------------
        tokens, lengths = tokens_dict['tokens'], \
            tokens_dict['lengths']
        tokens = tokens[:, : lengths.max()]

        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model. output is (batch_size, sequence_length, hidden_size)
        word_embeddings = self.encoder(tokens, mask).last_hidden_state

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()
                                             ).float().sum(1)
        sentemb = sentemb / sum_mask

        # Classification head
        logits = self.classification_head(sentemb)

        #-------------------------
        # DESCRIPTION EMBEDDINGS WITH GENERAL ATTENTION
        #-------------------------

        # # _process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        # k = self._process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)

        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.static_desc_emb:
        #     self.desc_emb = self._process_tokens(self.desc_tokens, type_as_tensor=k)[:, 0, :].squeeze(dim=1)

        # q = self.desc_emb.clone().type_as(k).expand(k.size(0), self.desc_emb.size(0), self.desc_emb.size(1))

        # # For Multiheadattention. Permuting because batch_size should be in dim=1
        # # for self.label_attn.
        # attn_output, _ = self.label_attn(
        #     q.transpose(1, 0), k.transpose(1, 0), k.transpose(1, 0)
        # )   # (num_classes, batch_size, hidden_dim)

        # logits = self.classification_head(attn_output).squeeze(dim=2)
        # logits = logits.transpose(1, 0)    # (batch_size, num_classes)

        #-------------------------------------------
        # DESCRIPTION EMBEDDINGS WITH MULTIHEAD BLOCKS
        #-------------------------------------------

        # # _process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        # x = self._process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)
        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.static_desc_emb:
        #     self.desc_emb = self._process_tokens(self.desc_tokens, type_as_tensor=x)[:, 0, :].squeeze(dim=1)

        # desc_emb = self.desc_emb.clone().type_as(x).expand(x.size(0), self.desc_emb.size(0), self.desc_emb.size(1))

        # # (batch_size, seq_len, hidden_dim)
        # output = desc_emb
        # for mod in self.label_attn:
        #     output = mod(x, output)

        # output = self.classification_head(output)
        # logits = self.final_fc.weight.mul(output).sum(dim=2).add(self.final_fc.bias)

        #---------------------------------
        # TRANSFORMERS
        #---------------------------------

        # # _process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        # x = self._process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)
        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.static_desc_emb:
        #     self.desc_emb = self._process_tokens(self.desc_tokens, type_as_tensor=x)[:, 0, :].squeeze(dim=1)

        # desc_emb = self.desc_emb.clone().type_as(x)\
        #     .expand(x.size(0), self.desc_emb.size(0), self.desc_emb.size(1))

        # # (batch_size, seq_len, hidden_dim)
        # output = desc_emb
        # for mod in self.label_attn:
        #     output = mod(x, output)

        # output = self.classification_head(output)
        # logits = self.final_fc.weight.mul(output).sum(dim=2).add(self.final_fc.bias)

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
