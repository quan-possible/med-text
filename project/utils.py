# -*- coding: utf-8 -*-
import json

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """

    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn

class F1WithLogitsLoss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, logits, labels):
        # assert y_hat.ndim == 2
        # assert y.ndim == 1
        # logits (batch_size, num_labels)
        # labels (batch_size, num_labels)
        y_hat = torch.sigmoid(logits).float()
        y = labels.float()

        # y = F.one_hot(y, 2).to(torch.float32)
        # y_hat = F.softmax(y_hat, dim=1)
        
        tp = (y * y_hat).sum(dim=0)
        tn = ((1 - y) * (1 - y_hat)).sum(dim=0)
        fp = ((1 - y) * y_hat).sum(dim=0)
        fn = (y * (1 - y_hat)).sum(dim=0)

        # precision = tp / (tp + fp + self.epsilon)
        # recall = tp / (tp + fn + self.epsilon)

        f1_1 = 2*tp / (2*tp + fn + fp + self.epsilon)
        f1_1 = 1 - f1_1.clamp(min=self.epsilon, max=1-self.epsilon)
        f1_0 = 2*tn / (2*tn + fn + fp + self.epsilon)
        f1_0 = 1 - f1_0.clamp(min=self.epsilon, max=1-self.epsilon)
        cost = (0.5 * (f1_1+f1_0)).mean()
        
        return cost

    
        
        

def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)

def parse_dataset_name(dataset: str):
    dataset = dataset.strip().lower()
    
    err_msg = "Invalid dataset name!"
    assert dataset in ['hoc', 'mtc'], err_msg
    
    return dataset

# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
