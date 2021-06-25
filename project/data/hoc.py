import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from nltk.corpus import stopwords
from string import digits
import os
import logging
from pathlib import Path
import random
from math import isclose

class HoC(object):

    def __init__(self, source_dir, target_dir, split=None):
        """
        source_dir: str or posix path. source directory of raw data
        target_dir: str or posix path. target directory to save split data
        split: tuple(int,int,int). train/val/test split proportions
        """
        assert isclose(sum(split), 1.0), "split proportions do not add up to 1."
        
        self.source_dir = Path(source_dir) if type(
            source_dir) is str else source_dir
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir
        self.split = split

        self.texts = self.read_texts(self.source_dir / 'text')
        self.labels = self.read_labels(self.source_dir / 'labels')
        
        target_files = set(
            [e.name for e in self.target_dir.iterdir() if e.is_file()])
        
        # check if there is existing file and split proportions
        if not set(['hoc_train.csv', 'hoc_val.csv', 'hoc_test.csv']) \
            .issubset(target_files) and split:
            self.split_and_save(self.texts, self.labels, self.target_dir, self.split)
        elif not set(['hoc.csv']).issubset(target_files):
            self.save_csv(self.texts, self.labels, self.target_dir)
            
    def split_and_save(self, texts, labels, target_dir, split):
        pd.DataFrame(np.array([texts[:int(len(texts) * split[0])],
                               labels[:int(len(texts) * split[0])]]).T,
                     columns=['text', 'labels']).to_csv(target_dir / 'hoc_train.csv', sep='\t')
        pd.DataFrame(np.array([texts[:int(len(texts) * split[1])], 
                               labels[:int(len(texts) * split[1])]]).T, 
                     columns=['text', 'labels']).to_csv(target_dir / 'hoc_val.csv', sep='\t')
        pd.DataFrame(np.array([texts[:int(len(texts) * split[2])], 
                               labels[:int(len(texts) * split[2])]]).T, 
                     columns=['text', 'labels']).to_csv(target_dir / 'hoc_test.csv', sep='\t')
    
    def save_csv(self, texts, labels, target_dir):
        logging.info("Saving data...")
        pd.DataFrame(np.array([texts, labels]).T, columns=[
                     'text', 'labels']).to_csv(target_dir / 'hoc.csv', sep='\t')

    def read_texts(text_dir):
        files = [x for x in text_dir.iterdir() if x.is_file()]
        texts = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())

        return texts

    def read_labels(label_dir):
        label_files = [x for x in label_dir.iterdir() if x.is_file()]
        all_labels = []
        for label_file in label_files:
            with open(label_file) as f:
                contents = f.read()
            all_labels.append(contents)

        sample_labels = []
        for label in all_labels:
            l = label.replace('<', '').strip().split('--')
            ll = []
            for i in l:
                try:
                    strings = i.strip().split()[0]
                    last_upper = 0
                    count = 0
                    for s in strings:
                        if s.isupper():
                            last_upper = count
                        count += 1
                    if last_upper == 0:
                        ll.append(strings)
                    else:
                        ll.append(strings[:last_upper])
                except:
                    ll.append('')
            sample_labels.extend(ll)

        sample_labels = list(set(sample_labels))
        
        data_labels = []
        for label in all_labels:
            data_labels.append(random.choice(
                label.replace('<', '').strip().split('--')).strip())

        labels = []
        for i in data_labels:
            try:
                strings = i.strip().split()[0]
                last_upper = 0
                count = 0
                for s in strings:
                    if s.isupper():
                        last_upper = count
                    count += 1
                if last_upper == 0:
                    labels.append(sample_labels.index(strings))
                else:
                    labels.append(sample_labels.index(strings[:last_upper]))
            except:
                labels.append(sample_labels.index(''))

        return labels
    
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
            help="Dataset chosen. HoC or MTC-5.",
        )
        parser.add_argument(
            "--num_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
    

    """ DEPRECATED """

    # def save_csv(self, target_dir):
    #     self.read_texts()
    #     self.read_labels()
    #     pd.DataFrame(np.array([texts, labels]).T, columns=[
    #                  'TEXT', 'LABEL']).to_csv('./data/hoc.csv', sep='\t')
    #     pd.DataFrame(np.array([texts[:int(len(texts)*0.2)], labels[:int(
    #         len(texts)*0.2)]]).T, columns=['TEXT', 'LABEL']).to_csv('./data/hoc_test.csv', sep='\t')
    #     pd.DataFrame(np.array([texts[:int(len(texts)*0.1)], labels[:int(
    #         len(texts)*0.1)]]).T, columns=['TEXT', 'LABEL']).to_csv('./data/hoc_val.csv', sep='\t')


if __name__ == "__main__":
    # testing
    hoc = HoC("./project/data/HoC", "./project/data", (0.7,0.15,0.15))
