import pathlib
from pathlib import Path
from argparse import ArgumentParser, Namespace
from sys import path

import pandas as pd
import torch
import yaml

from classifier import Classifier
from tokenizer import Tokenizer
from datamodule import DataModule, Collator
from utils import dotdict

from torchmetrics.functional import f1, precision_recall
from collections import OrderedDict



def load_hparams(experiment_dir: str):
    hparams_file = experiment_dir + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)
    
    return dotdict(hparams)


def load_model(experiment_dir: str, hparams, tokenizer, collator, num_classes):
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
    
    model = Classifier.load_from_checkpoint(
        checkpoint_path, hparams=hparams, tokenizer=tokenizer,
        num_classes=num_classes,
        collator=collator, encoder_model=hparams.encoder_model,
        batch_size=hparams.batch_size,
        nr_frozen_epochs=hparams.nr_frozen_epochs,
        #  label_encoder,
        encoder_learning_rate=hparams.encoder_learning_rate, 
        learning_rate=hparams.learning_rate,
    )
    
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model

def main(args):
    print("Loading model...")
    hparams = load_hparams(args.experiment_dir)
    
    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = DataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, hparams.num_workers,
        hparams.tgt_txt_col, hparams.tgt_lbl_col,
    )
    
    num_classes = datamodule.num_classes

    model = load_model(args.experiment_dir, hparams, tokenizer,
                       collator, num_classes)
    
    datamodule.setup()
    
    # TODO: model interaction
    
    # cuda = torch.device('cuda')     # Default CUDA device
    
    # test_dataloader = datamodule.test_dataloader()
    # acc_ls = []
    # f1_ls = []
    # precision_ls = []
    # recall_ls = []
    
    # model.cuda()
    
    # for batch in test_dataloader:
    #     inputs, targets = batch
        
    #     inputs = {key: value.cuda() for (key, value) in inputs.items()}
    #     targets = {key: value.cuda() for key, value in targets.items()}
        
    #     model_out = model.forward(inputs)

    #     y = targets["labels"]
    #     logits = model_out["logits"]

    #     preds = torch.argmax(logits, dim=1)

    #     # acc
    #     test_acc = torch.sum(y == preds).type_as(y) / (len(y) * 1.0)
    #     acc_ls.append(test_acc)

    #     # f1
    #     test_f1 = f1(preds, y, num_classes=num_classes, average='macro')
    #     f1_ls.append(test_f1)

    #     # precision and recall
    #     test_precision, test_recall = precision_recall(
    #         preds, y, num_classes=num_classes, average='macro')
    #     precision_ls.append(test_precision)
    #     recall_ls.append(test_recall)

    #     # metrics = OrderedDict({"val_acc": test_acc, "val_f1": test_f1,
    #     #                        "val_precision": test_precision,
    #     #                        "val_recall": test_recall})
    
    # reduce_ = lambda x: sum(x)/len(x)
    
    # print(
    #     reduce_(acc_ls),
    #     reduce_(f1_ls),
    #     reduce_(precision_ls),
    #     reduce_(recall_ls),
    # )   

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to the experiment folder.",
    )
    
    args = parser.parse_args()
    
    main(args)

    

    


