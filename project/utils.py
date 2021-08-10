# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from decimal import Decimal, getcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
from torch import Tensor

class F1Loss(nn.Module):
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

    def __init__(self, sigmoid=False, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.sigmoid = sigmoid
        
    def forward(self, logits, labels):
        if self.sigmoid:
            logits = torch.sigmoid(logits)
            
        y_hat = logits.float()
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
    
class TextAttentionHeatmap(object):
    def __init__(self, target_dir='nice.tex', color='red') -> None:
        super().__init__()
        self.color = color
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir
        self.configure_latex()
        
    def configure_latex(self):
        with open(self.target_dir, 'w') as f:
            f.write(
                r'''
\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}
                ''' + '\n'
            )
    
    def __call__(self, text, attn, rescale=False):
        assert len(text) == len(attn)
        if rescale:
            attn = self._rescale()
        num_words = len(text)
        text_cleaned = self._clean_text(text)
        with open(self.target_dir,'w') as f:
            colored_text = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{''' + "\n"
            for idx in range(num_words):
                colored_text += "\\colorbox{%s!%s}{"%(self.color, attn[idx])+"\\strut " + text_cleaned[idx]+"} "
            colored_text += r"\n}}}"
            f.write(colored_text + "\n")
            f.write(r'''\end{CJK*}
\end{document}''')
            
    def _rescale(self, attn):
        attn_arr = np.asarray(attn)
        max_ = np.max(attn_arr)
        min_ = np.min(attn_arr)
        rescale = (attn_arr - min_)/(max_-min_)*100
        return rescale.tolist()
    
    def _clean_text(text):
        new_word_list = []
        for word in text:
            for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
                if latex_sensitive in word:
                    word = word.replace(latex_sensitive, '\\'+latex_sensitive)
            new_word_list.append(word)
        return new_word_list
        


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
    

        

def parse_dataset_name(dataset: str, options=['hoc', 'mtc']):
    dataset = dataset.strip().lower()
    
    err_msg = "Invalid dataset name!"
    assert dataset in options, err_msg
    
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



