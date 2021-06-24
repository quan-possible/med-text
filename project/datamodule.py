# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from math import floor
from random import random
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers import AutoModel

import pytorch_lightning as pl
from .tokenizer import Tokenizer
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
# from utils import mask_fill


class Collator(object):
    
    def __init__(self, tokenizer, prepare_targets: bool = True) -> None:
        """
        Object that prepares a sample to input the model when called.
        :param tokenizer: Tokenizer class instance.
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.prepare_targets = prepare_targets

    def __call__(self, sample):
        """
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not self.prepare_targets:
            return inputs, {}

        targets = sample["labels"]
        return inputs, targets

class DataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 classifier_instance, tokenizer,
                 data_path: str, batch_size, num_workers,
                 txt_col_name="TEXT", lbl_col_name="LABEL"):
        super().__init__()
        
        self.data_path = Path(data_path) if type(
            data_path) is str else data_path

        self.classifier_instance = classifier_instance
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lbl_col_name = lbl_col_name
        self.txt_col_name = txt_col_name
    
    def prepare_data(self):
        pass
        
    
    def setup(self):
        self.collator = Collator(self.tokenizer)
        self._train_dataset = self._read_csv(self.data_path / "hoc_train.csv")
        self._val_dataset = self._read_csv(self.data_path / "hoc_val.csv")
        self._test_dataset = self._read_csv(self.data_path / "hoc_test.csv")
        

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the test set. """
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.loader_workers,
        )
        
    @staticmethod
    def _read_csv(data_path: str, txt_col_name="TEXT",
                  lbl_col_name="LABEL") -> list:
        """ Reads a comma separated value file.

        :param path: path to a csv file.
        
        :return: List of records as dictionaries
        """
        df = pd.read_csv(data_path, sep='\t', index_col=0,)
        df[txt_col_name] = df[txt_col_name].astype(str)
        df[lbl_col_name] = df[lbl_col_name]
        return df.to_dict("records")

    @staticmethod
    def add_model_specific_args(parent_parser):
        return


        

""" DEPRECATED """

# class MedDataset(Dataset):
#     def __init__(self, dataset, txt_col_name="TEXT",
#                     lbl_col_name="LABEL"):
#         self.dataset = dataset
#         self.lbl_col_name = lbl_col_name
#         self.txt_col_name = txt_col_name

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         text = self.dataset[idx][self.txt_col_name]
#         label = self.dataset[idx][self.lbl_col_name]

#         return text, label

