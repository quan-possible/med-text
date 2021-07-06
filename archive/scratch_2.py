# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Tuple


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel

import pytorch_lightning as pl
from torchmetrics.functional import f1, precision_recall
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import lengths_to_mask
from pytorch_lightning.utilities.seed import seed_everything
from utils import mask_fill, dotdict


class Classifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams, tokenizer, collator, encoder_model,
                 batch_size, num_classes, nr_frozen_epochs,
                 #  label_encoder,
                 encoder_learning_rate, learning_rate,
                 ) -> None:
        super(Classifier, self).__init__()

        # self.hparams = hparams
        self.tokenizer = tokenizer
        self.collator = collator

        self.batch_size = batch_size
        self.num_classes = num_classes
        # self.label_encoder = label_encoder
        self.encoder_model = encoder_model
        self.encoder_learning_rate = encoder_learning_rate
        self.learning_rate = learning_rate

        self.save_hyperparameters(
            # "batch_size",
            # "nr_frozen_epochs", "encoder_learning_rate",
            # "learning rate",
            hparams,
        )

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.bert = AutoModel.from_pretrained(
            self.encoder_model, output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        if self.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.num_classes),
        )

    def forward(self, tokens_lengths):
        """ Usual pytorch forward function. 
        :param tokens_lengths: tuple of:
            - text sequences [batch_size x src_seq_len]
            - lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens, lengths = tokens_lengths['tokens'], \
            tokens_lengths['lengths']
        tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch

        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        # (batch_size, sequence_length, hidden_size)
        word_embeddings = self.bert(tokens, mask).last_hidden_state

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()
                                             ).float().sum(1)
        sentemb = sentemb / sum_mask

        # Classification head
        logits = self.classification_head(sentemb)

        return {"logits": logits}

    def __build_loss(self):
        """ Initializes the loss function/s. """
        if self.dataset == 'hoc':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.collator(
                [sample], prepare_target=False)
            model_out = self.forward(model_input)
            logits = model_out["logits"].numpy()
            # predicted_labels = [
            #     self.data.label_encoder.index_to_token[prediction]
            #     for prediction in np.argmax(logits, axis=1)
            # ]
            # sample["predicted_label"] = predicted_labels[0]

            sample["predicted_label"] = np.argmax(logits, axis=1)[0]

        return sample

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x num_classes] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self.loss_fn(predictions["logits"], targets["labels"])

    def training_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_idx: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss = self.loss(model_out, targets)

        # acc
        y = targets["labels"]
        logits = model_out["logits"]
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(y == preds).type_as(y) / (len(y) * 1.0)

        self.log("loss", loss)
        self.log("acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss = self.loss(model_out, targets)

        y = targets["labels"]
        logits = model_out["logits"]

        preds = torch.argmax(logits, dim=1)

        # acc
        val_acc = torch.sum(y == preds).type_as(y) / (len(y) * 1.0)

        # f1
        val_f1 = f1(preds, y, num_classes=self.num_classes, average='macro')

        # precision and recall
        val_precision, val_recall = precision_recall(
            preds, y, num_classes=self.num_classes, average='macro')

        loss_acc = OrderedDict({"val_loss": loss, "val_acc": val_acc})
        metrics = OrderedDict({"val_f1": val_f1, "val_precision": val_precision,
                              "val_recall": val_recall})

        self.log_dict(loss_acc, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

        # # can also return just a scalar instead of a dict (return loss_val)
        return loss_acc

    def test_step(self, batch: tuple, batch_idx: int, *args, **kwargs):
        inputs, targets = batch
        model_out = self.forward(inputs)

        y = targets["labels"]
        logits = model_out["logits"]

        preds = torch.argmax(logits, dim=1)

        # acc
        test_acc = torch.sum(y == preds).type_as(y) / (len(y) * 1.0)

        # f1
        test_f1 = f1(preds, y, num_classes=self.num_classes, average='macro')

        # precision and recall
        test_precision, test_recall = precision_recall(
            preds, y, num_classes=self.num_classes, average='macro')

        metrics = OrderedDict({"val_acc": test_acc, "val_f1": test_f1,
                               "val_precision": test_precision,
                               "val_recall": test_recall})
        self.log_dict(metrics)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.encoder_learning_rate,
            },
        ]
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)
        return [self.optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) \
            -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """

        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )

        return parser

    " DEPRECATED "
    # def validation_epoch_end(self, outputs: list) -> dict:
    #     """ Function that takes as input a list of dictionaries returned by the validation_step
    #     function and measures the model performance accross the entire validation set.

    #     Returns:
    #         - Dictionary with metrics to be added to the lightning logger.
    #     """
    #     val_loss_mean = 0
    #     val_acc_mean = 0
    #     for output in outputs:
    #         val_loss = output["val_loss"]

    #         # reduce manually when using dp
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_loss = torch.mean(val_loss)
    #         val_loss_mean += val_loss

    #         # reduce manually when using dp
    #         val_acc = output["val_acc"]
    #         if self.trainer.use_dp or self.trainer.use_ddp2:
    #             val_acc = torch.mean(val_acc)

    #         val_acc_mean += val_acc

    #     val_loss_mean /= len(outputs)
    #     val_acc_mean /= len(outputs)
    #     tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
    #     result = {
    #         "progress_bar": tqdm_dict,
    #         "log": tqdm_dict,
    #         "val_loss": val_loss_mean,
    #         "val_acc": val_acc_mean,
    #     }


    #     return result
if __name__ == "__main__":
    # ENCODER_MODEL = "bert-base-uncased"
    # DATA_PATH = "./project/data"
    # DATASET = "hoc"
    # BATCH_SIZE = 2
    # NUM_WORKERS = 2
    # NR_FROZEN_EPOCHS = 1
    # ENCODER_LEARNING_RATE = 1e-05
    # LEARNING_RATE = 3e-05

    seed_everything(69)

    hparams = dotdict(
        encoder_model="bert-base-cased",
        data_path="./project/data",
        dataset="mtc",
        batch_size=2,
        num_workers=2,
        random_sampling=False,
        nr_frozen_epochs=1,
        encoder_learning_rate=1e-05,
        learning_rate=3e-05,
        tgt_txt_col="TEXT",
        tgt_lbl_col="LABEL",
    )

    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, hparams.num_workers,
        # hparams.tgt_txt_col, hparams.tgt_lbl_col,
    )

    num_classes = datamodule.num_classes

    model = Classifier(
        hparams, tokenizer, collator, hparams.encoder_model,
        hparams.batch_size, num_classes, hparams.nr_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
