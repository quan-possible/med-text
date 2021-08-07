from base_classifier import BaseClassifier
from multilabel import MultiLabelClassifier
from multiclass import MultiClassClassifier
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator

import pathlib
import yaml
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


def load_model(experiment_dir: str, dataset, hparams, tokenizer, collator, num_classes):
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
    
    classifier = HOCClassifier if dataset == "hoc" else MTCClassifier
    
    model = classifier.load_from_checkpoint(
        checkpoint_path, hparams=hparams, tokenizer=tokenizer,
        collator=collator, encoder_model=hparams.encoder_model,
        batch_size=hparams.batch_size,
        num_frozen_epochs=hparams.num_frozen_epochs,
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
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, hparams.num_workers,
    )
    
    num_classes = datamodule.num_classes

    model = load_model(args.experiment_dir, hparams.dataset, 
                       hparams, tokenizer,
                       collator, num_classes)
    
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

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to the experiment folder.",
    )
    
    args = parser.parse_args()
    
    main(args)

    

    


