# -*- coding: utf-8 -*-
import json

from datetime import datetime
from pathlib import Path
from decimal import Decimal, getcontext
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
from torch import Tensor


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """

    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.out_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query: Tensor, value: Tensor):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn

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

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn
    
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

def calc_scheduler_lr(
    lr, num_warmup_steps, num_training_steps, num_frozen_epochs, scheduler_epochs,
    steps_per_epoch=17.296
):
    num_frozen_steps = steps_per_epoch * num_frozen_epochs
    total_num_steps = steps_per_epoch * scheduler_epochs
    def encoder_lr_lambda(current_step: int):
        if current_step >= num_frozen_steps:
            current_step = current_step - num_frozen_steps
        else:
            return 0.0
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        res = round(float(num_training_steps - current_step) /
                      float(max(1, num_training_steps - num_warmup_steps)), 8)
        return max(0.0, res)
    mark = total_num_steps // 6
    mark1, mark2, mark3 = mark * 3, mark * 4, mark * 5
    res = {
        "unfreeze_step": num_frozen_steps,
        "end_step": total_num_steps,
        "step300_lr": lr * encoder_lr_lambda(300),
        "step" + str(mark1) + "_lr": lr * encoder_lr_lambda(mark1),
        "step" + str(mark2) + "_lr": lr * encoder_lr_lambda(mark2),
        "step" + str(mark3) + "_lr": lr * encoder_lr_lambda(mark3),
        "final_lr": lr * encoder_lr_lambda(total_num_steps)
    }
        
    return res

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__