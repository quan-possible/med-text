# -*- coding: utf-8 -*-
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from base_classifier import BaseClassifier
from utils import F1WithLogitsLoss, AdditiveAttention, DotProductAttention


from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, OrderedDict

import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics.functional import accuracy, f1, precision, recall, precision_recall
from pytorch_lightning.utilities.seed import seed_everything

class HOCClassifier(BaseClassifier):

    def __init__(
        self, hparams, desc_tokens, tokenizer, collator, encoder_model, 
        batch_size, nr_frozen_epochs, encoder_learning_rate, learning_rate, 
        num_heads, static_desc_emb=True,
    ):
        super().__init__(
            hparams, desc_tokens, tokenizer, collator, encoder_model,
            batch_size, nr_frozen_epochs, encoder_learning_rate, 
            learning_rate, num_heads,
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

        if nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
            
        self.nr_frozen_epochs = nr_frozen_epochs
    
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
            average=self.hparams.metric_averaging
        )
        
        # precision and recall
        precision_, recall_ = precision_recall(
            normed_logits, labels, num_classes=self.num_classes,
            average=self.hparams.metric_averaging
        )
        
        return acc, f1_, precision_, recall_, p_class_score
    
    
    def _build_model(self, encoder_model) -> None:
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
            
        # set the number of features our encoder model will return...
        self._label_attn = nn.MultiheadAttention(
            self.encoder_features, self.num_heads, dropout=0.2,
        )

        # self.label_attn = DotProductAttention(self.encoder_features)
        
        # Classification head
        self._classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, 1),
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
        k = self._process_tokens(tokens_dict) # (batch_size, seq_len, hidden_dim)
        
        # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        if not self.static_desc_emb:
            self.desc_emb = self._process_tokens(self.desc_tokens, type_as_tensor=k)[:, 0, :].squeeze()
        
        q = self.desc_emb.type_as(k).expand(k.size(0), self.desc_emb.size(0), self.desc_emb.size(1))
        
        # random init
        # q = self.desc_emb.type_as(k).expand(k.size(0), self.desc_emb.size(0), self.desc_emb.size(1)).detach()  # (batch_size, num_classes, hidden_dim)
        
        # attn_output, _ = self.label_attn(k, k)
        
        # For Multiheadattention. Permuting because batch_size should be in dim=1 
        # for self.label_attn.
        attn_output, _ = self.label_attn(
            q.transpose(1, 0), k.transpose(1, 0), k.transpose(1, 0)
        )   # (num_classes, batch_size, hidden_dim) 
        
        logits = self.classification_head(attn_output).squeeze()
        logits = logits.transpose(1, 0)    # (batch_size, num_classes)

        return {"logits": logits}
    
    def _detach_dict(self, tensor_dict):
        return {k:v.detach() for k,v in tensor_dict.items()}
        
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
        num_heads=8,
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
    
    desc_tokens = datamodule.desc_tokens
    # print(desc_tokens)
    print("Load description finished!")

    model = HOCClassifier(
        hparams, desc_tokens, tokenizer, collator, 
        hparams.encoder_model,
        hparams.batch_size, hparams.nr_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
        hparams.num_heads,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
