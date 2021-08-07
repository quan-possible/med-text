import numpy as np
import pandas as pd
# from nltk.stem import WordNetLemmatizer
import re
# from nltk.corpus import stopwords
from string import digits
from pathlib import Path
import logging


class MTC(object):

    def __init__(self, source_dir, target_dir, update_existing=True, split=[0.8, 0.1, 0.1]):
        """
        source_dir: str or posix path. source directory of raw data
        target_dir: str or posix path. target directory to save split data
        """
        
        self.source_dir = Path(source_dir) if type(
            source_dir) is str else source_dir
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir
        
        self.split = split
        self.texts, self.labels = self.read_raw_data(self.source_dir / 'train.dat')
        # print(self.labels)

        # self.texts = self.preprocess_text(self.texts)

        target_files = set(
            [e.name for e in self.target_dir.iterdir() if e.is_file()])
        
        if update_existing:
            self.split_and_save(self.texts, self.labels, self.target_dir, self.split) if self.split\
                else self.save_csv(self.texts, self.labels, self.target_dir)
        else:
            if not set(['mtc_train.csv', 'mtc_val.csv', 'mtc_test.csv']) \
                    .issubset(target_files) and self.split:
                self.split_and_save(self.texts, self.labels, self.target_dir, self.split)
            elif not set(['mtc.csv']).issubset(target_files):
                self.save_csv(self.texts, self.labels, self.target_dir)

    def preprocess_text(self, texts):
    #     # texts_filtered = self._filterLen(
    #     #     [l.split() for l in texts], 4)  # TODO
    #     # texts_processed = self._text_preprocess(texts_filtered)
        texts_processed = [text.replace('\n', '') for text in texts]

        return texts_processed

    def split_and_save(self, texts, labels, target_dir, split):
        assert split != None, "No split proportions available."

        pd.DataFrame(np.array([texts[:int(len(texts) * split[0])],
                               labels[:int(len(texts) * split[0])]]).T,
                     columns=['TEXT', 'LABEL']).to_csv(target_dir / 'mtc_train.csv', sep='\t')
        pd.DataFrame(np.array([texts[:int(len(texts) * split[1])],
                               labels[:int(len(texts) * split[1])]]).T,
                     columns=['TEXT', 'LABEL']).to_csv(target_dir / 'mtc_val.csv', sep='\t')
        pd.DataFrame(np.array([texts[:int(len(texts) * split[2])],
                               labels[:int(len(texts) * split[2])]]).T,
                     columns=['TEXT', 'LABEL']).to_csv(target_dir / 'mtc_test.csv', sep='\t')

    def save_csv(self, texts, labels, target_dir):
        logging.info("Saving data...")
        pd.DataFrame(np.array([texts, labels]).T, columns=[
            'TEXT', 'LABEL']).to_csv(target_dir / 'mtc.csv', sep='\t')

    # # remove short words
    # def _filterLen(self, tdocs, minlen):
    #     return [' '.join(t for t in d if len(t) >= minlen) for d in tdocs]

    # # lemmatize and remove stop words
    # def _text_preprocess(self, data_arr):
    #     lemmatiser = WordNetLemmatizer()  # TODO
    #     pattern = re.compile(
    #         r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    #     # print train_arr[0:10]
    #     for i in range(len(data_arr)):
    #         data_arr[i] = pattern.sub('', data_arr[i])
    #         data_arr[i] = data_arr[i].translate(digits)  # TODO
    #     return data_arr

    @staticmethod
    def read_raw_data(data_file):
        text, labels = [], []
        with open(data_file, "r") as fh:
            lines = fh.readlines()

        for i in range(len(lines)):
            text.append(lines[i][2:])
            labels.append(int(lines[i][0:1]))
        labels = np.asarray(labels) - 1 # Map to base-0
        return text, labels

    """ DEPRECATED """

    # def split_and_save(self, ratio=[0.8, 0.1, 0.1]):
    #     """
    #     ration: list of a triplet. train/val/test ratio. default: [0.8, 0.1, 0.1]
    #     """
    #     txt_filtered = self._filterLen(
    #         [l.split() for l in self.txt], 4)  # TODO
    #     self.txt = self._text_preprocess(txt_filtered)
    #     txt_train, txt_test, lbl_train, lbl_test = train_test_split(
    #         self.txt, self.lbl, test_size=1-ratio[0])
    #     txt_val, txt_test, lbl_val, lbl_test = train_test_split(
    #         txt_test, lbl_test, test_size=ratio[2]/(ratio[2]+ratio[1]))
    #     logging.info(f"Number of train samples: {len(lbl_train)}")
    #     logging.info(f"Number of valid samples: {len(lbl_val)}")
    #     logging.info(f"Number of test samples: {len(lbl_test)}")

    #     pd.DataFrame(np.array([txt_train, lbl_train]).T, columns=['TEXT', 'LABEL']).to_csv(
    #         self.target_dir/'train.csv', sep='\t', index=False)
    #     pd.DataFrame(np.array([txt_val, lbl_val]).T, columns=['TEXT', 'LABEL']).to_csv(
    #         self.target_dir/'val.csv', sep='\t', index=False)
    #     pd.DataFrame(np.array([txt_test, lbl_test]).T, columns=['TEXT', 'LABEL']).to_csv(
    #         self.target_dir/'test.csv', sep='\t', index=False)


if __name__ == "__main__":
    mtc = MTC("./project/data/MTC-5/data", "./project/data")
