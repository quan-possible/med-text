# -*- coding: utf-8 -*-
from datetime import datetime

import torch

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
        # assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # logits (batch_size, num_labels)
        # labels (batch_size, num_labels)
        y_hat = torch.sigmoid(logits).float()
        y = labels.float()

        # y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

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
