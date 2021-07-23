# -*- coding: utf-8 -*-
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from base_classifier import BaseClassifier
from label_attention import LabelAttentionLayer
from utils import F1WithLogitsLoss, AdditiveAttention, DotProductAttention

import copy
import numpy as np
from math import floor
from argparse import Namespace
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics.functional import accuracy, f1, precision, \
    recall, precision_recall
from pytorch_lightning.utilities.seed import seed_everything


class HOCClassifier(BaseClassifier):

    def __init__(
        self, hparams, desc_tokens, tokenizer, collator, encoder_model,
        batch_size, num_frozen_epochs, encoder_learning_rate, learning_rate,
        num_heads, num_warmup_steps, num_training_steps, 
        metric_averaging, static_desc_emb=True,
    ):
        super().__init__(
            hparams, desc_tokens, tokenizer, collator, encoder_model,
            batch_size, num_frozen_epochs, encoder_learning_rate,
            learning_rate, num_heads, num_warmup_steps, num_training_steps, 
            num_frozen_epochs, metric_averaging,
        )

        self._num_classes = 10

        # build model
        self._build_model(self.encoder_model)

        # Loss criterion initialization.
        self._build_loss()

        self.desc_tokens = desc_tokens  # (batch_size, seq_len)

        self.static_desc_emb = static_desc_emb
        if self.static_desc_emb:
            with torch.no_grad():
                self.desc_emb = self._process_tokens(self.desc_tokens)[:, 0, :].squeeze()

        if num_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        self.num_frozen_epochs = num_frozen_epochs

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def label_attn(self):
        return self._label_attn

    @property
    def encoder(self):
        return self._encoder

    @property
    def classification_head(self):
        return self._classification_head
    
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _build_loss(self):
        self._loss_fn = nn.BCEWithLogitsLoss()
        # self._loss_fn = F1WithLogitsLoss()

    def get_metrics(self, logits, labels):
        normed_logits = torch.sigmoid(logits)

        # acc
        acc = accuracy(normed_logits, labels)

        # per class metrics
        p_class_score = torch.zeros((self.num_metrics, self.num_classes))
        p_class_f = [f1, precision, recall]
        for y in range(self.num_classes):
            for x, f in enumerate(p_class_f):
                p_class_score[x, y] = f(logits[:, y], labels[:, y])

        # f1
        f1_ = f1(
            normed_logits, labels, num_classes=self.num_classes,
            average=self.metric_averaging
        )

        # precision and recall
        precision_, recall_ = precision_recall(
            normed_logits, labels, num_classes=self.num_classes,
            average=self.metric_averaging
        )

        return acc, f1_, precision_, recall_, p_class_score

    def _build_model(self, encoder_model, n_lbl_attn_layer=3) -> None:
        """ Init BERT model + tokenizer + classification head."""

        if encoder_model == "google/bigbird-pegasus-large-pubmed":
            self.encoder_features = 1024
            self._encoder = AutoModel.from_pretrained(
                encoder_model, attention_type="original_full", output_hidden_states=True
            )
        else:
            self.encoder_features = 768
            self._encoder = AutoModel.from_pretrained(
                encoder_model, output_hidden_states=True
            )
            
        
        filter_sizes = [3,5,7] # Only choose odd numbers
        self.dim_conv = int(768/len(filter_sizes))
        
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.encoder_features,
                      out_channels=self.dim_conv,
                      kernel_size=size,
                      padding=floor(size / 2))
            for size in filter_sizes
        ])
        self.bn1d_list = nn.ModuleList([nn.BatchNorm1d(self.dim_conv) for size in filter_sizes])
        
        # label_attn_layer = LabelAttentionLayer(self.encoder_features, self.num_heads)
        # self._label_attn = self._get_clones(label_attn_layer, n_lbl_attn_layer)
        # # self.label_attn = DotProductAttention(self.encoder_features)
        
        # self.trans1 = nn.Transformer(
        #     d_model=self.encoder_features, nhead=12, num_decoder_layers=2,
        #     num_encoder_layers=2, batch_first=True,
        # )
        
        # self.trans2 = nn.Transformer(
        #     d_model=self.encoder_features, nhead=12, num_decoder_layers=2,
        #     num_encoder_layers=2, batch_first=True,
        # )

        # Classification head
        self._classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.num_classes),
        )
        

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        return self._loss_fn(predictions["logits"], targets["labels"].float())

    def forward(self, tokens_dict):
        """ Usual pytorch forward function.
            :param tokens_dict: tuple of:
                - text sequences [batch_size x src_seq_len]
                - lengths: source lengths [batch_size]

            Returns:
                Dictionary with model logit outputs of size 
                (batch_size, num_classes) 
            """
        # _process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        x = self._process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)
        
        x = x.transpose(2, 1)  # (batch_size, hidden_dim, seq_len)
        
        x_list = [F.relu(bn1d(conv1d(x))) for conv1d, 
             bn1d in zip(self.conv1d_list, self.bn1d_list)]
        
        x_list = [F.max_pool1d(x, kernel_size=x.size(2)) for x in x_list]

        x = torch.cat([x.squeeze(dim=2) for x in x_list], dim=1)
        
        logits = self.classification_head(x).squeeze()
        
        # src = torch.cat(x, dim=1).transpose(2, 1)  # (batch_size, seq_len, hidden_dim)
        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.static_desc_emb:
        #     self.desc_emb = self._process_tokens(self.desc_tokens, type_as_tensor=src)[:, 0, :].squeeze()

        # key_value = self.desc_emb.clone().type_as(src)\
        #     .expand(src.size(0), self.desc_emb.size(0), self.desc_emb.size(1))
        #     # (batch_size, num_classes, hidden_dim)

        # # For Multiheadattention. Permuting because batch_size should be in dim=1
        # # for self.label_attn.
        # # attn_output, _ = self.label_attn(
        # #     q.transpose(1, 0), k.transpose(1, 0), k.transpose(1, 0)
        # # )   # (num_classes, batch_size, hidden_dim)
        
        # # (batch_size, seq_len, hidden_dim)
        # output = src
        # for mod in self.label_attn:
        #     output = mod(output, key_value)
        
        # src = self._process_tokens(tokens_dict)

        # if not self.static_desc_emb:
        #     self.desc_emb = self._process_tokens(self.desc_tokens, type_as_tensor=src)[:, 0, :].squeeze()
            
        # tgt = self.desc_emb.clone().type_as(src)\
        #     .expand(src.size(0), self.desc_emb.size(0), self.desc_emb.size(1))
            
        # tgt = self.trans1(src, tgt) 
        # tgt = self.trans2(src, tgt)

        # print(output.size())
        # CLS pooling
        # attn_output = output[:,0,:]
        
        # logits = self.classification_head(tgt).squeeze()  # (batch_size, num_classes)
        # logits = logits.transpose(1, 0)

        return {"logits": logits}

    def _detach_dict(self, tensor_dict):
        return {k: v.detach() for k, v in tensor_dict.items()}

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
    
    
def proto_main():
    
    seed_everything(69)

    hparams = Namespace(
        encoder_model="bert-base-cased",
        data_path="./project/data",
        dataset="hoc",
        batch_size=2,
        num_workers=2,
        random_sampling=False,
        num_frozen_epochs=0,
        encoder_learning_rate=1e-05,
        learning_rate=3e-05,
        num_heads=8,
        tgt_txt_col="TEXT",
        tgt_lbl_col="LABEL",
        metric_averaging="micro",
        num_warmup_steps=50,
        num_training_steps=100,
    )

    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size,
        hparams.num_workers,
    )

    desc_tokens = datamodule.desc_tokens
    # print(desc_tokens)
    print("Load description finished!")

    model = HOCClassifier(
        hparams, desc_tokens, tokenizer, collator,
        hparams.encoder_model,
        hparams.batch_size, hparams.num_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
        hparams.num_heads, hparams.num_warmup_steps,
        hparams.num_training_steps, hparams.metric_averaging,
    )
    
    return hparams, tokenizer, collator, datamodule, model


if __name__ == "__main__":

    hparams, tokenizer, collator, datamodule, model = proto_main()

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
