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
        y_hat = torch.sigmoid(logits).float()
        y = labels.float()
        
        tp = (y * y_hat).sum(dim=0)
        tn = ((1 - y) * (1 - y_hat)).sum(dim=0)
        fp = ((1 - y) * y_hat).sum(dim=0)
        fn = (y * (1 - y_hat)).sum(dim=0)

        f1_1 = 2*tp / (2*tp + fn + fp + self.epsilon)
        f1_1 = 1 - f1_1.clamp(min=self.epsilon, max=1-self.epsilon)
        f1_0 = 2*tn / (2*tn + fn + fp + self.epsilon)
        f1_0 = 1 - f1_0.clamp(min=self.epsilon, max=1-self.epsilon)
        cost = (0.5 * (f1_1+f1_0)).mean()
        
        return cost


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


def get_lr_schedule(
    param_groups, encoder_indices: list, optimizer,
    scheduler_epochs, num_frozen_epochs, steps_per_epoch,
    warmup_pct=[0.1, 0.1], smallest_lr_pct=[0.005, 0.005, 0.4],
    #     num_warmup_steps, num_training_steps, num_frozen_epochs, scheduler_epochs=10,
    #     steps_per_epoch=1, smallest_lr_pct=0.05,
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    num_frozen_steps = steps_per_epoch * num_frozen_epochs
    total_num_steps = steps_per_epoch * scheduler_epochs
    num_unfrozen_steps = total_num_steps - num_frozen_steps
    num_warmup_steps = num_unfrozen_steps * warmup_pct[0]
    beta = 0.1 * total_num_steps

    def encoder_lr_lambda(current_step: int):
        res = smallest_lr_pct[0]

        if current_step < num_frozen_steps:
            return res
        else:
            current_step = current_step - num_frozen_steps

        if current_step < num_warmup_steps:
            res = float(current_step) / float(max(1, num_warmup_steps))
        else:
            res = float(num_unfrozen_steps - current_step) \
                / float(max(1, num_unfrozen_steps - num_warmup_steps))

        return max(smallest_lr_pct[0], res)
    
    total_num_steps = steps_per_epoch * scheduler_epochs
    num_warmup_steps = total_num_steps * warmup_pct[1]
    
    def lbl_attn_lambda(current_step: int):
        if current_step < num_warmup_steps:
            res = float(current_step) / float(max(1, num_warmup_steps))
        else:
            res = float(total_num_steps - current_step) \
                / float(max(1, total_num_steps - num_warmup_steps))

        return max(smallest_lr_pct[1], res)

    def normal_lr_lambda(current_step: int):
        return max(smallest_lr_pct[1], 1.0 - (current_step / (total_num_steps + beta)))

    # lambda_list = [encoder_lr_lambda if idx in encoder_indices
    #                else normal_lr_lambda for idx in range(len(param_groups))]
    lambda_list = [encoder_lr_lambda, lbl_attn_lambda, normal_lr_lambda]

    return LambdaLR(optimizer, lambda_list)

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
    return embeddings.float().masked_fill_(\
        padding_mask, fill_value).type_as(embeddings)

def parse_dataset_name(dataset: str):
    dataset = dataset.strip().lower()
    
    err_msg = "Invalid dataset name!"
    assert dataset in ['hoc', 'mtc'], err_msg
    
    return dataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
