from base_classifier import BaseClassifier
from multilabel import MultiLabelClassifier
from multiclass import MultiClassClassifier
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator

import pathlib
import yaml
import numpy as np
import pandas as pd
from sys import path
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torchnlp.random import set_seed
from torchmetrics.functional import f1, precision_recall

class TextAttentionHeatmap(object):
    def __init__(self, target_dir='attn_heatmap.tex', color='red') -> None:
        super().__init__()
        self.color = color
        self.target_dir = Path(target_dir) if type(
            target_dir) is str else target_dir
        self.configure_latex()
        
    def __call__(self, text, attn, rescale=True):
        assert len(text) == len(attn)
        if rescale:
            attn = self._rescale(attn)
        num_words = len(text)
        text_cleaned = self._clean_text(text)
        with open(self.target_dir, 'a') as f:
            colored_text = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{''' + "\n"
            for idx in range(num_words):
                colored_text += "\\colorbox{%s!%s}{" % (self.color, attn[idx]) + "\\strut " + text_cleaned[idx] + "} "
            colored_text += r"\n}}}"
            f.write(colored_text + "\n")
            f.write(r'''\end{CJK*}
\end{document}''')
        
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
            
    def _rescale(self, attn):
        attn_arr = np.asarray(attn)
        max_ = np.max(attn_arr)
        min_ = np.min(attn_arr)
        rescale = ((attn_arr - min_)/(max_-min_)*100).tolist()
        
        return ["{: .8f}".format(float(str(alpha))) for alpha in rescale]
    
    def _clean_text(self, text):
        new_word_list = []
        for word in text:
            for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
                if latex_sensitive in word:
                    word = word.replace(latex_sensitive, '\\'+latex_sensitive)
            new_word_list.append(word)
        return new_word_list


def prototype(hparams):
    seed_everything(69)
    
    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, 
        hparams.num_workers,
    )
    
    desc_tokens = datamodule.desc_tokens
    num_classes = datamodule.num_classes
    train_size = datamodule.size(dim=0)
    print("Finished loading data!")
    
    
    if hparams.dataset == 'hoc':
        model = MultiLabelClassifier(
            desc_tokens, tokenizer, collator, num_classes, train_size, hparams, **vars(hparams)
        )
    else:
        model = MultiClassClassifier(
            desc_tokens, tokenizer, collator, num_classes, train_size, hparams, **vars(hparams)
        )
        
    return model, datamodule, hparams

def load_hparams(experiment_dir: str):
    hparams_file = experiment_dir + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)
    # print(Namespace(**hparams))
    return Namespace(**hparams)


def load_model(experiment_dir: str, desc_tokens, tokenizer, collator, 
               num_classes, train_size, hparams):
    """ Function that loads the model from an experiment folder.
    :param experiment_dir: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    experiment_path = Path(experiment_dir + "/checkpoints/")
    
    # hparams_file = experiment_dir + "/hparams.yaml"
    # hparams = dotdict(yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader))

    checkpoints = [
        file.name
        for file in experiment_path.iterdir()
        if file.name.endswith(".ckpt")
    ]
    checkpoint_path = experiment_path / checkpoints[-1]
    
    if hparams.dataset == 'hoc':
        model = MultiLabelClassifier(
            desc_tokens, tokenizer, collator, num_classes, train_size, hparams, **vars(hparams)
        )
    else:
        model = MultiClassClassifier(
            desc_tokens, tokenizer, collator, num_classes, train_size, hparams, **vars(hparams)
        )
    
    model = model.load_from_checkpoint(
        checkpoint_path, 
        desc_tokens=desc_tokens, 
        tokenizer=tokenizer, 
        collator=collator,
        num_classes=num_classes, 
        train_size=train_size, 
        hparams=hparams
    )

    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model

def main(args):
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Loading model...")
    hparams = load_hparams(args.experiment_dir)
    
    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, hparams.num_workers,
    )
    
    desc_tokens = datamodule.desc_tokens
    num_classes = datamodule.num_classes
    train_size = datamodule.size(dim=0)

    print("Finished loading data")
    model = load_model(args.experiment_dir, desc_tokens, tokenizer, 
                       collator, num_classes, train_size, hparams)
    
    print("All parameters:")
    for name, _ in model.named_parameters():
        print(name)
    
    print("Please wait for processing...")
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    
    model.cuda()
    metrics = OrderedDict({"test_acc": 0,
                        "test_f1": 0,
                        "test_precision": 0,
                        "test_recall": 0})
    
    for batch in test_dataloader:
        inputs, targets = batch
        
        inputs = {key: value.cuda() for (key, value) in inputs.items()}
        targets = {key: value.cuda() for key, value in targets.items()}
        
        model_out = model.forward(inputs)

        labels = targets["labels"]
        logits = model_out["logits"]

        test_acc, test_f1, test_precision, test_recall = \
            model._get_metrics(logits, labels)
            
        metrics["test_acc"] += test_acc
        metrics["test_f1"] += test_f1
        metrics["test_precision"] += test_precision
        metrics["test_recall"] += test_recall
        
        
    metrics = {key: value/len(test_dataloader) for (key, value) in metrics.items()}

    print(metrics)
    
def interpret(args, document):
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    heatmap_generator = TextAttentionHeatmap()

    print("Loading model...")
    hparams = load_hparams(args.experiment_dir)

    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, hparams.num_workers,
    )

    desc_tokens = datamodule.desc_tokens
    num_classes = datamodule.num_classes
    train_size = datamodule.size(dim=0)
    
    print("Finished loading data")
    
    model = load_model(args.experiment_dir, desc_tokens, tokenizer,
                       collator, num_classes, train_size, hparams)
    
    y_pred = model.predict(document)[0]
    print("Prediction:")
    print(y_pred)
    # print(y_pred.item())
    
    doc_dict = {"text": document}
    doc_tokens,_ = collator(doc_dict, prepare_targets=False)
    # print(doc_tokens["tokens"].size())
    doc_words = tokenizer.tokenizer.convert_ids_to_tokens(doc_tokens["tokens"][0])
    
    # print(doc_tokens)
    x = model.process_tokens(doc_tokens)
    desc_emb = model.process_tokens(desc_tokens, type_as_tensor=x)[:, 0, :].squeeze(dim=1)
    desc_emb = desc_emb.type_as(x).expand(x.size(0), desc_emb.size(0), desc_emb.size(1))
    
    x = x.transpose(0, 1)
    desc_emb = desc_emb.transpose(0,1)
    _, attn_output_weights = model.label_attn[0].lbl_attn(desc_emb, x, x)
    x = x.transpose(0, 1)
    desc_emb = desc_emb.transpose(0, 1)
    
    
    print(x.size())
    print(desc_emb.size())
    print(attn_output_weights.size())
    
    attn = attn_output_weights[0,y_pred.item(),:].tolist()
    print("attn shape:")
    print(len(attn))
    print("====")
    print(len(doc_words))
    
    heatmap_generator(doc_words, attn)

if __name__ == "__main__":
    
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to the experiment folder.",
    )
    
    args = parser.parse_args()
    
    # # check test metrics
    # main(args)
    document = "Renal abscess in children. Three cases of renal abscesses in children are described to illustrate the variable presenting features. An additional 23 pediatric cases, reported over the past ten years, were reviewed for clinical features and therapy. Fever, loin pain, and leukocytosis were common presenting features, but less than half of all abscesses were associated with either an abnormal urinalysis or a positive urine culture. The presenting features were sometimes confused with appendicitis, peritonitis, or a Wilms tumor. An organism was identified in 17 cases - -Escherichia coli in 9 children and Staphylococcus aureus in 8 children. The majority of E. coli infections occurred in girls and the majority of S. aureus infections occurred in boys. Reflux was documented in 5 patients, and 2 children had a possible extrarenal source of infection. Antibiotics alone produced a cure in 10 children (38 %), but 16 children (62%) required a surgical procedure."
    interpret(args, document)

    
    

    


