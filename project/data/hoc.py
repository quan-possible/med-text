import pandas as pd
import re

from pathlib import Path
from collections import defaultdict, OrderedDict


class HOC():
    def __init__(self, source_dir, target_dir, new_labels: dict, replace_csv=True) -> None:
        
        self.source_dir = Path(source_dir) if type(
            source_dir) is str else source_dir
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir
        self.labels = self.get_labels(source_dir)
        self.new_labels = [new_labels[label] for label in self.labels]
        self.replace_csv = replace_csv
        
        
        self.train_dict = self.read_dataset(source_dir, self.labels, 'train')
        self.val_dict = self.read_dataset(source_dir, self.labels, 'devel')
        self.test_dict = self.read_dataset(source_dir, self.labels, 'test')
            
        self.save_csv(self.train_dict, self.target_dir, 
                      self.new_labels, self.replace_csv, set_='train')
        self.save_csv(self.val_dict, self.target_dir, 
                      self.new_labels, self.replace_csv, set_='val')
        self.save_csv(self.test_dict, self.target_dir,
                      self.new_labels, self.replace_csv, set_='test')
    
    def save_csv(self, data_dict, target_dir, labels, 
                 replace_csv, set_):
        target_files = set(
            [e.name for e in target_dir.iterdir() if e.is_file()])
        filename = f"hoc_{set_}.csv"

        if (not replace_csv and filename not in target_files) \
            or replace_csv:
                
            pd.DataFrame.from_dict(
                data_dict, orient='index', columns=labels
            ).to_csv(target_dir / filename, sep='\t')


    def read_dataset(self, source_dir, labels, set_='train'):
        pattern = set_ + '.*'
        data_dict = defaultdict(lambda: [])

        for file in source_dir.rglob(pattern):
            self._read_file(file, data_dict, file.parent.name)

        data_dict = {k: self._one_hot(v, labels) for k, v in data_dict.items()}

        return OrderedDict(sorted(data_dict.items()))


    def get_labels(self, source_dir, pattern=r"label-."):
        labels = []
        for dir_ in source_dir.iterdir():
            if dir_.is_dir() and re.match(pattern, dir_.name):
                labels.append(dir_.name)

        return sorted(labels)


    def _read_file(self, file, data_dict: defaultdict, label):
        pos = re.match(".*\.pos", file.name)
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    processed_line = line.strip().replace('\n', '')
                    if pos:
                        data_dict[processed_line].append(label)
                    else:
                        data_dict[processed_line]
        return

    def _one_hot(self, target_labels, all_labels):
        return [int(x in target_labels) for x in all_labels]
        
if __name__ == "__main__":
    SOURCE_DIR, TARGET_DIR = Path("project\data\HoC"), Path("project\data")
    REPLACE_CSV = True
    NEW_LABELS = {
        "label-1": "Activating invasion and metastasis",
        "label-2": "Avoiding immune destruction",
        "label-3": "Cellular energetics",
        "label-4": "Enabling replicative immortality",
        "label-5": "Evading growth suppressors",
        "label-6": "Genomic instability and mutation",
        "label-7": "Inducing angiogenesis",
        "label-8": "Resisting cell death",
        "label-9": "Sustaining proliferative signaling",
        "label-a": "Tumor promoting inflammation",
    }
    
    hoc = HOC(SOURCE_DIR, TARGET_DIR, NEW_LABELS, REPLACE_CSV)
