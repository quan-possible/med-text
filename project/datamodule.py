# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from math import floor
from random import random
from typing import Tuple
from pathlib import Path
import json

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
from pytorch_lightning.utilities.seed import seed_everything
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
# from utils import mask_fill


class Collator(object):
    
    def __init__(self, tokenizer) -> None:
        """
        Object that prepares a sample to input the model when called.
        :param tokenizer: Tokenizer class instance.
        """

        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, sample, prepare_targets: bool = True):
        """
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_targets:
            return inputs, {}

        targets = {"labels": torch.tensor(np.array(sample["labels"]))}
        return inputs, targets


class MedDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, collator, data_path: str, dataset,
                 batch_size, num_workers):
        super().__init__()

        self.data_path = Path(data_path) if type(
            data_path) is str else data_path
        self.dataset = dataset.strip().lower()

        self.tokenizer = tokenizer
        self.collator = collator

        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.labels = None
        # self._desc_tokens = None
        
        if self.dataset == 'hoc':
            self.read_csv = self.read_hoc
            self.labels = pd.read_csv(self.data_path / f'{self.dataset}_train.csv', \
                sep='\t', index_col=0, nrows=0).columns.tolist()
            
        else:
            self.read_csv = self.read_mtc
            self.labels = ['0','1','2','3','4']
            
        self._num_classes = len(self.labels)
        self._desc_tokens = self.read_desc(
            self.dataset, self.data_path, self.labels,
        )
            
    @property
    def desc_tokens(self):
        return self._desc_tokens
        
    @property
    def num_classes(self):
        return self._num_classes
    
    def setup(self, stage=None):
        if stage in (None, "fit"):
            self._train_dataset = self.read_csv(self.data_path /
                                                f"{self.dataset}_train.csv")
            self._val_dataset = self.read_csv(self.data_path /
                                              f"{self.dataset}_val.csv")

        if stage in (None, 'test'):
            self._test_dataset = self.read_csv(self.data_path /
                                               f"{self.dataset}_test.csv")

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self._train_dataset,
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
            num_workers=self.num_workers,
        )
        
    # def read_hoc_desc(self, dataset, data_path, labels):
    #     with open(data_path / "hoc_labels.json", 'r') as f:
    #         desc_dict = json.load(f)
    #     desc = {"text": [desc_dict[label] for label in labels]}
    #     desc_tokens, _ = self.collator(desc, prepare_targets=False)
    #     return desc_tokens
    
    def read_desc(self, dataset, data_path, labels):
        with open(data_path / f"{dataset}_labels.json", 'r') as f:
            desc_dict = json.load(f)

        desc = {"text": [desc_dict[label] for label in labels]}
        desc_tokens, _ = self.collator(desc, prepare_targets=False)
        return desc_tokens
        
    @classmethod
    def read_dataset_name(cls, dataset):
        dataset = dataset.strip().lower()

    @classmethod
    def read_hoc(cls, file_path: str) -> list:
        """ Reads a comma separated value file.

        :param path: path to a csv file.
        
        :return: 
            - List of records as dictionaries
            - Number of classes
        """
        df = pd.read_csv(file_path, sep='\t', index_col=0,)
        df["labels"] = list(df.values)
        df["text"] = df.index.values.astype(str)

        return df[['text', 'labels']].to_dict("records")

    @classmethod
    def read_mtc(
        cls, file_path: str,
    ) -> list:
        """ Reads a comma separated value file.

        :param path: path to a csv file.
        
        :return: 
            - List of records as dictionaries
            - Number of classes
        """
        df = pd.read_csv(file_path, sep='\t', index_col=0,)
        df["text"] = df["TEXT"].astype(str).str.replace("\n","")
        df["labels"] = df["LABEL"]

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
            default="hoc",
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

        return parser


if __name__ == "__main__":
    
    seed_everything(69)
    
    hparams = Namespace(
        encoder_model="bert-base-cased",
        data_path="./project/data",
        dataset="mtc",
        batch_size=2,
        num_workers=2,
        random_sampling=False,
        num_frozen_epochs=1,
        encoder_learning_rate=1e-05,
        learning_rate=3e-05,
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
    
    print(datamodule.desc_tokens)
    
    # datamodule.setup()
    
    # print(next(iter(datamodule.train_dataloader())))
    
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

