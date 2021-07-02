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
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers import AutoModel

import pytorch_lightning as pl
from tokenizer import Tokenizer
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

        # print(type(sample["labels"]))
        # print(inputs["tokens"].size())
        targets = {"labels": torch.tensor(sample["labels"])}
        return inputs, targets

class DataModule(pl.LightningDataModule):
    
    def __init__(self, tokenizer, collator, data_path: str, dataset,
                 batch_size, num_workers, rand_sampling=True):
        super().__init__()
        
        self.data_path = Path(data_path) if type(
            data_path) is str else data_path
        self.dataset = dataset.strip().lower()

        self.tokenizer = tokenizer
        self.collator = collator
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tgt_txt_col, self.tgt_lbl_col = "TEXT", "LABEL"
        
        self.sampler = RandomSampler if rand_sampling \
            else SequentialSampler
        
        self.n_classes = 36 if self.dataset == 'hoc' else 5

    
    def prepare_data(self):
        pass
        
    
    def setup(self, stage=None):
        if stage in (None, "fit"):
            self._train_dataset = self.read_csv(self.data_path / 
                                                f"{self.dataset}_train.csv", 
                                                self.dataset)
            self._val_dataset = self.read_csv(self.data_path / 
                                            f"{self.dataset}_val.csv", 
                                            self.dataset)
            
        if stage in (None, 'test'):
            self._test_dataset = self.read_csv(self.data_path / 
                                                f"{self.dataset}_test.csv",
                                                self.dataset)
        

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self._train_dataset,
            sampler=self.sampler(self._train_dataset),
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
    def read_csv(file_path: str, dataset,
                 tgt_txt_col="TEXT", 
                 tgt_lbl_col="LABEL") -> list:
        """ Reads a comma separated value file.

        :param path: path to a csv file.
        
        :return: 
            - List of records as dictionaries
            - Number of classes
        """
        df = pd.read_csv(file_path, sep='\t', index_col=0,)
        df["text"] = df[tgt_txt_col].astype(str)
        df["labels"] = df[tgt_lbl_col] if dataset == 'hoc' \
            else df[tgt_lbl_col] - 1
        
        return df[['text', 'labels']].to_dict("records")

    @staticmethod
    def add_model_specific_args(parser):
        
        parser.add_argument(
            "--data_path",
            default="./project/data",
            type=str,
            help="Path to directory containing the data.",
        )
        parser.add_argument(
            "--dataset",
            default="mtc",
            type=str,
            help="Dataset chosen. 'hoc' (HoC) or 'mtc' (MTC-5).",
        )
        parser.add_argument(
            "--num_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        
        parser.add_argument(
            "--txt_col_name",
            default="TEXT",
            type=str,
            help="Column name of the texts in the csv files.",
        )
        
        parser.add_argument(
            "--lbl_col_name",
            default="LABEL",
            type=str,
            help="Column name of the labels in the csv files.",
        )
        
        return parser


if __name__ == "__main__":
    
    MODEL = "bert-base-cased"
    DATA_PATH = "./project/data"
    DATASET = "mtc"
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    RANDOM_SAMPLING = False
    
    tokenizer = Tokenizer(MODEL)
    collator = Collator(tokenizer)
    datamodule = DataModule(
        tokenizer, collator, DATA_PATH, 
        DATASET, BATCH_SIZE, NUM_WORKERS, RANDOM_SAMPLING
    )
    
    datamodule.setup()
    
    print(next(iter(datamodule.train_dataloader())))
    
    ### TODO: Write unit test
    
    
    

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

