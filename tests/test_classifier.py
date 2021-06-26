import sys
sys.path.insert(0, 'D:/OneDrive - Aalto University/Courses/Thesis/med-text')
# TODO: Find a better way.

import torch

from pytorch_lightning import seed_everything
from project.classifier import Classifier
from project.datamodule import DataModule, Collator
from project.tokenizer import Tokenizer


def test_forward(batch_size):
    
    ENCODER_MODEL = "bert-base-uncased"
    DATA_PATH = "./project/data"
    DATASET = "hoc"
    NUM_WORKERS = 2
    NR_FROZEN_EPOCHS = 1
    ENCODER_LEARNING_RATE = 1e-05
    LEARNING_RATE = 3e-05
    RANDOM_SAMPLING = False

    seed_everything(69)

    tokenizer = Tokenizer(ENCODER_MODEL)
    collator = Collator(tokenizer)
    datamodule = DataModule(
        tokenizer, collator, DATA_PATH,
        DATASET, batch_size, NUM_WORKERS,
        rand_sampling=RANDOM_SAMPLING,
    )

    n_classes = datamodule.n_classes

    model = Classifier(
        tokenizer, collator, ENCODER_MODEL,
        batch_size, n_classes, NR_FROZEN_EPOCHS,
        ENCODER_LEARNING_RATE, LEARNING_RATE,
    )
    
    datamodule.setup()
    inputs_samp, targets_samp = next(iter(datamodule.train_dataloader()))
    res = model(**inputs_samp)

    err_msg = "Incorrect output shape!"
    assert res['logits'].size() == torch.Size([batch_size, n_classes]), err_msg
    
if __name__ == "__main__":
    
    BATCH_SIZE = 2
    test_forward(BATCH_SIZE)
