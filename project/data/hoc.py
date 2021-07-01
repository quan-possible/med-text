import logging
import numpy as np
import pandas as pd
import re
import pathlib
import typing

from pathlib import Path
from collections import defaultdict, OrderedDict


class HoC():
    def __init__(self, source_dir, target_dir) -> None:
        
        self.source_dir = Path(source_dir) if type(
            source_dir) is str else source_dir
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir
        self.labels = self.get_labels(source_dir)
        
        self.train_dict = self.read_dataset(source_dir, self.labels, 'train')
        self.val_dict = self.read_dataset(source_dir, self.labels, 'devel')
        self.test_dict = self.read_dataset(source_dir, self.labels, 'test')
            
        self.save_csv(self.train_dict, self.target_dir, self.labels, set_='train')
        self.save_csv(self.val_dict, self.target_dir, self.labels, set_='val')
        self.save_csv(self.test_dict, self.target_dir, self.labels, set_='test')
    

    def save_csv(self, data_dict, target_dir, labels, set_='train'):
        target_files = set(
            [e.name for e in target_dir.iterdir() if e.is_file()])
        filename = f"hoc_{set_}.csv"

        if filename not in target_files:
            pd.DataFrame.from_dict(
                data_dict, orient='index', columns=labels
            ).to_csv(target_dir / filename)


    def read_dataset(self, source_dir, labels, set_='train'):
        pattern = set_ + '.pos'
        data_dict = defaultdict(lambda: [])

        for file in source_dir.rglob(pattern):
            self._read_file(file, data_dict, file.parent.name)

        data_dict = {k: self._one_hot(v, labels) for
                    k, v in data_dict.items()}

        return OrderedDict(sorted(data_dict.items()))


    def get_labels(self, source_dir, pattern=r"label-."):
        labels = []
        for dir_ in source_dir.iterdir():
            if dir_.is_dir() and re.match(pattern, dir_.name):
                labels.append(dir_.name)

        return sorted(labels)


    def _read_file(self, file, data_dict: defaultdict, label):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    data_dict[line].append(label)
        return

    def _one_hot(self, target_labels, all_labels):
        return [int(x in target_labels)
                for x in all_labels]
        
if __name__ == "__main__":
    data = Path("project\data\HoC")
    hoc = HoC(data, data)
