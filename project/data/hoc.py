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


class HoC(object):


    def __init__(self, source_dir, target_dir):
        """
        source_dir: str or posix path. source directory of raw data
        target_dir: str or posix path. target directory to save split data
        """
        self.source_dir = Path(source_dir) if type(
            source_dir) is str else source_dir
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir

        self.texts = self.read_texts(self.source_dir / 'text')
        self.labels = self.read_labels(self.source_dir / 'labels')
        
        target_files = set(
            [e.name for e in self.target_dir.iterdir() if e.is_file()])
        if not set(['hoc.csv']).issubset(target_files):
            self.save_csv(self.texts, self.labels, self.target_dir)
        
    def save_csv(self, texts, labels, target_dir):
        logging.info("Saving data...")
        pd.DataFrame(np.array([texts, labels]).T, columns=[
                     'TEXT', 'LABEL']).to_csv(target_dir / 'hoc.csv', sep='\t')
        
    @staticmethod
    def read_texts(text_dir):
        files = [x for x in text_dir.iterdir() if x.is_file()]
        texts = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())

        return texts

    @staticmethod
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

    """ DEPRECATED """

    #
    # def save_csv(self, target_dir):
    #     self.read_texts()
    #     self.read_labels()
    #     pd.DataFrame(np.array([self.train_arr, self.train_lbl]).T, columns=[
    #                  'TEXT', 'LABEL']).to_csv('./data/hoc.csv', sep='\t')
    #     pd.DataFrame(np.array([self.train_arr[:int(len(self.train_arr)*0.2)], self.train_lbl[:int(
    #         len(self.train_arr)*0.2)]]).T, columns=['TEXT', 'LABEL']).to_csv('./data/hoc_test.csv', sep='\t')
    #     pd.DataFrame(np.array([self.train_arr[:int(len(self.train_arr)*0.1)], self.train_lbl[:int(
    #         len(self.train_arr)*0.1)]]).T, columns=['TEXT', 'LABEL']).to_csv('./data/hoc_val.csv', sep='\t')


if __name__ == "__main__":
    # testing
    hoc = HoC("./project/data/HoC", "./project/data")
