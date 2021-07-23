from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from base_classifier import BaseClassifier
from utils import F1WithLogitsLoss, AdditiveAttention, DotProductAttention

import numpy as np
from argparse import Namespace
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LabelAttentionLayer(nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (default=12).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
    """

    def __init__(self, d_model, nhead=12, dim_feedforward=2048, 
            dropout=0.1, batch_first=True, 
        ) -> None:
        super(LabelAttentionLayer, self).__init__()
        self.batch_first = batch_first
        self.lbl_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        layer_norm_eps = 1e-05
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, desc_emb, desc_src=True):
        r"""Pass the input through the label attention layer.

            Args:
                src: the sequence to the encoder layer (required).
                src_mask: the mask for the src sequence (optional).
                src_key_padding_mask: the mask for the src keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
        if self.batch_first:
            src = src.transpose(0,1)
            desc_emb = desc_emb.transpose(0,1)
        
        src2, _ = self.lbl_attn(desc_emb, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if self.batch_first:
            src = src.transpose(0, 1)
            
        return src
