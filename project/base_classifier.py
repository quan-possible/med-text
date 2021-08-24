# -*- coding: utf-8 -*-
from tokenizer import Tokenizer
from datamodule import MedDataModule, Collator
from label_attention import LabelAttentionLayer

import copy
import logging as log
from math import ceil,floor
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from transformers import AutoModel
from torchnlp.utils import lengths_to_mask
from pytorch_lightning.utilities.seed import seed_everything
from utils import mask_fill, get_lr_schedule, str2bool


class BaseClassifier(pl.LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, desc_tokens, tokenizer, collator, num_classes, train_size, hparams, *args, **kwargs) -> None:
        super(BaseClassifier, self).__init__()

        self.desc_tokens = desc_tokens  # (batch_size, seq_len)
        self.tokenizer = tokenizer
        self.collator = collator
        self.num_classes = num_classes
        self.train_size = train_size

        self.save_hyperparameters(hparams)
        
        # build model
        self._build_model()

        # Loss criterion initialization.
        self._build_loss()

        if self.hparams.static_desc_emb:
            with torch.no_grad():
                self.desc_emb = self.process_tokens(self.desc_tokens)[:, 0, :].squeeze(dim=1)

        if self.hparams.num_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        

    @abstractmethod
    def _get_metrics(self, logits, labels):
        pass

    @abstractmethod
    def _build_loss(self):
        pass
    
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def predict(self, sample: dict):
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        pass

    def _build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        # pass

        self.encoder = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            # hidden_dropout_prob=0.1,
            output_hidden_states=True,
        )

        # set the number of features our encoder model will return...
        if self.hparams.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768
            
        #----------------------------
        # CONVOLUTIONAL HEAD
        #----------------------------
        
        # filter_sizes = [3, 5, 7]  # Only choose odd numbers
        # self.dim_conv = int(self.encoder_features / len(filter_sizes))
        
        # # Conv Network
        # self.conv1d_list = nn.ModuleList([
        #     nn.Conv1d(in_channels=self.encoder_features,
        #               out_channels=self.dim_conv,
        #               kernel_size=size,
        #               padding=floor(size / 2))
        #     for size in filter_sizes
        # ])
        # self.bn1d_list = nn.ModuleList([nn.BatchNorm1d(self.dim_conv) for _ in filter_sizes])

        #----------------------------
        # LABEL ATTENTION
        #----------------------------
        if self.hparams.n_lbl_attn_layer > 0:
            label_attn_layer = LabelAttentionLayer(self.encoder_features)
            self.label_attn = self._get_clones(label_attn_layer, self.hparams.n_lbl_attn_layer)

        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
        )
        
        self.final_fc = nn.Linear(self.encoder_features, self.num_classes)

    def process_tokens(self, tokens_dict, type_as_tensor=None):
        tokens, lengths = tokens_dict['tokens'], \
            tokens_dict['lengths']
        tokens = tokens[:, : lengths.max()]

        if type_as_tensor != None:
            tokens = tokens.to(type_as_tensor.device).detach()

        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch. In other words, use this when using DataParallel.
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        # output is (batch_size, sequence_length, hidden_size)
        emb = self.encoder(tokens, mask).last_hidden_state

        return emb

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    def forward(self, tokens_dict):
        """ Usual pytorch forward function. 
        :param tokens_dict: tuple of:
            - text sequences [batch_size x src_seq_len]
            - lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        #----------------------------
        # CONVOLUTIONAL HEAD
        #----------------------------
        
        # # process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        # x = self.process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)
        # # print(x.size())
        # # print("nice")
        
        # x = x.transpose(2, 1)  # (batch_size, hidden_dim, seq_len)

        # x_list = [F.relu(bn1d(conv1d(x))) for conv1d,
        #           bn1d in zip(self.conv1d_list, self.bn1d_list)]

        # # x_list = [F.max_pool1d(x, kernel_size=x.size(2)) for x in x_list]
        # # print(x_list[0].size())
        # x = torch.cat([x for x in x_list], dim=1)
        
        # x = x.transpose(2, 1)
        # # print(x.size())

        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.hparams.static_desc_emb:
        #     self.desc_emb = self.process_tokens(self.desc_tokens, type_as_tensor=x)[:, 0, :].squeeze(dim=1)

        # desc_emb = self.desc_emb.clone().type_as(x).expand(x.size(0), self.desc_emb.size(0), self.desc_emb.size(1))
        
        # output = desc_emb
        # for mod in self.label_attn:
        #     output = mod(x, output)
            
        # output = self.classification_head(output)
        # logits = self.final_fc.weight.mul(output).sum(dim=2).add(self.final_fc.bias)
            
        #----------------------------
        # NO HEAD
        #----------------------------
        # tokens, lengths = tokens_dict['tokens'], \
        #     tokens_dict['lengths']
        # tokens = tokens[:, : lengths.max()]

        # # When using just one GPU this should not change behavior
        # # but when splitting batches across GPU the tokens have padding
        # # from the entire original batch
        # mask = lengths_to_mask(lengths, device=tokens.device)

        # # Run BERT model. output is (batch_size, sequence_length, hidden_size)
        # word_embeddings = self.encoder(tokens, mask).last_hidden_state

        # # Average Pooling
        # word_embeddings = mask_fill(
        #     0.0, tokens, word_embeddings, self.tokenizer.padding_index
        # )
        # sentemb = torch.sum(word_embeddings, 1)
        # sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()
        #                                      ).float().sum(1)
        # sentemb = sentemb / sum_mask

        # # Classification head
        # logits = self.classification_head(sentemb)
        # logits = self.final_fc(logits)
        
        #-------------------------
        # DESCRIPTION EMBEDDINGS WITH GENERAL ATTENTION
        #-------------------------
        
        # # process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        # k = self.process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)

        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.hparams.static_desc_emb:
        #     self.desc_emb = self.process_tokens(self.desc_tokens, type_as_tensor=k)[:, 0, :].squeeze(dim=1)

        # q = self.desc_emb.clone().type_as(k).expand(k.size(0), self.desc_emb.size(0), self.desc_emb.size(1))

        # # For Multiheadattention. Permuting because batch_size should be in dim=1
        # # for self.label_attn.
        # attn_output, _ = self.label_attn(
        #     q.transpose(1, 0), k.transpose(1, 0), k.transpose(1, 0)
        # )   # (num_classes, batch_size, hidden_dim)

        # logits = self.classification_head(attn_output).squeeze(dim=2)
        # logits = logits.transpose(1, 0)    # (batch_size, num_classes)
        
        #-------------------------------------------
        # DESCRIPTION EMBEDDINGS WITH MULTIHEAD BLOCKS
        #-------------------------------------------
        
        # process_tokens is defined in BaseClassifier. Simply input the tokens into BERT.
        x = self.process_tokens(tokens_dict)  # (batch_size, seq_len, hidden_dim)
        # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        if self.hparams.n_lbl_attn_layer > 0:
            if not self.hparams.static_desc_emb:
                self.desc_emb = self.process_tokens(self.desc_tokens, type_as_tensor=x)[:, 0, :].squeeze(dim=1)

            desc_emb = self.desc_emb.clone().type_as(x).expand(x.size(0), self.desc_emb.size(0), self.desc_emb.size(1))
            
            output = desc_emb
            for mod in self.label_attn:
                output = mod(x, output)
                
            output = self.classification_head(output)
            
            # print(output.size())
            logits = self.final_fc.weight.mul(output).sum(dim=2).add(self.final_fc.bias)
        
        else:
            output = x[:, 0, :]
            output = self.classification_head(output)
            logits = self.final_fc(output)
        
        # (batch_size, seq_len, hidden_dim)


        
        #---------------------------------
        # RANDOM
        #---------------------------------
        
        # tokens, lengths = tokens_dict['tokens'], \
        #     tokens_dict['lengths']
        # tokens = tokens[:, : lengths.max()]

        # # When using just one GPU this should not change behavior
        # # but when splitting batches across GPU the tokens have padding
        # # from the entire original batch
        # mask = lengths_to_mask(lengths, device=tokens.device)

        # # Run BERT model. output is (batch_size, sequence_length, hidden_size)
        # x = self.encoder(tokens, mask).last_hidden_state
        
        # # CLS pooling for label descriptions. output shape is (num_classes, hidden_dim)
        # if not self.hparams.static_desc_emb:
        #     self.desc_emb = self.process_tokens(self.desc_tokens, type_as_tensor=x)[:, 0, :].squeeze(dim=1)

        # desc_emb = self.desc_emb.clone().type_as(x).expand(x.size(0), self.desc_emb.size(0), self.desc_emb.size(1))

        # # (batch_size, seq_len, hidden_dim)
        # output = x
        # for mod in self.label_attn:
        #     output = mod(query=output, key=desc_emb)
            
        # # Average Pooling
        # word_embeddings = mask_fill(
        #     0.0, tokens, output, self.tokenizer.padding_index
        # )
        # sentemb = torch.sum(word_embeddings, 1)
        # sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()
        #                                      ).float().sum(1)
        # sentemb = sentemb / sum_mask

        # # Classification head
        # logits = self.classification_head(sentemb)
        # logits = self.final_fc(logits)


        return {"logits": logits}

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._frozen = True

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        encoder_names = ['encoder']
        label_attn = ['_label_attn']
        param_groups = [
            {
                'params': [p for n, p in self.named_parameters()
                           if any(nd in n for nd in encoder_names)],
                'name': "encoder",
                'lr': self.hparams.encoder_learning_rate,
                'weight_decay': self.hparams.weight_decay_encoder,
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if any(nd in n for nd in label_attn)],
                'name': "label_attention",
                'lr': self.hparams.label_attn_lr,
                'weight_decay': self.hparams.weight_decay_encoder,
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in (encoder_names + label_attn))],
                'name': 'non-encoder',
                'weight_decay': self.hparams.weight_decay_nonencoder,
            },
        ]

        self.optimizer = optim.AdamW(param_groups, lr=self.hparams.learning_rate)

        # crucial to converge
        steps_per_epoch = ceil(self.train_size / (self.hparams.batch_size * self.hparams.accumulate_grad_batches * self.hparams.gpus))
        self.lr_scheduler = get_lr_schedule(
            param_groups=param_groups, encoder_indices=[0], optimizer=self.optimizer,
            scheduler_epochs=self.hparams.scheduler_epochs,
            num_frozen_epochs=self.hparams.num_frozen_epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_pct=[self.hparams.warmup_pct, self.hparams.warmup_pct],
            smallest_lr_pct=[self.hparams.smallest_lr_pct_encoder, self.hparams.smallest_lr_pct_lbl_attn,
                             self.hparams.smallest_lr_pct_nonencoder],
        )

        # self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=[5e-05, 1e-03], epochs=self.hparams.max_epochs,
        #     steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='linear',
        #     cycle_momentum=False, div_factor=2.50, final_div_factor=20.0,
        #     three_phase=False, last_epoch=-1, verbose=True
        # )

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'interval': 'step',
            }
        }

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.hparams.num_frozen_epochs:
            self.unfreeze_encoder()

    def training_step(self, batch: tuple, batch_idx) -> dict:
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

        self.log("loss", loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        val_loss = self.loss(model_out, targets)

        labels = targets["labels"]   # (batch_size, num_labels)
        logits = model_out["logits"]    # (batch_size, num_labels)

        val_acc, val_f1, val_precision, val_recall = \
            self._get_metrics(logits, labels)

        loss_acc = OrderedDict({"val_loss": val_loss, "val_acc": val_acc})
        metrics = OrderedDict({"val_f1": val_f1,
                               "val_precision": val_precision,
                               "val_recall": val_recall})

        self.log_dict(loss_acc, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.log("hp_metric", val_f1)

        # # can also return just a scalar instead of a dict (return loss_val)
        return loss_acc

    def test_step(self, batch: tuple, batch_idx: int,) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(inputs)

        labels = targets["labels"]   # (batch_size, num_labels)
        logits = model_out["logits"]    # (batch_size, num_labels)

        test_acc, test_f1, test_precision, test_recall = \
            self._get_metrics(logits, labels)

        metrics = OrderedDict({"test_acc": test_acc,
                               "test_f1": test_f1,
                               "test_precision": test_precision,
                               "test_recall": test_recall})

        self.log_dict(metrics, sync_dist=True)

        # can also return just a scalar instead of a dict (return loss_val)
        return metrics

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
            "--num_frozen_epochs",
            default=10,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )

        parser.add_argument(
            "--metric_averaging",
            default="micro",
            type=str,
            help="Averaging methods for validation metrics (micro, macro,...)",
        )

        parser.add_argument(
            "--num_heads",
            default=12,
            type=int,
            help="Averaging methods for validation metrics (micro, macro,...)",
        )

        parser.add_argument(
            "--warmup_pct",
            default=0.1,
            type=float,
            help="Percentage of training steps used for warmup",
        )

        parser.add_argument(
            "--scheduler_epochs",
            default=30,
            type=int,
            help="Number of epochs the scheduler for the encoder is activated",
        )

        parser.add_argument(
            "--weight_decay_encoder",
            default=0.05,
            type=float,
            help="Weight decay for encoder",
        )

        parser.add_argument(
            "--weight_decay_nonencoder",
            default=0.1,
            type=float,
            help="Weight decay for non-encoder",
        )

        parser.add_argument(
            "--smallest_lr_pct_encoder",
            default=0.01,
            type=float,
            help="Smallest encoder learning rate being a percentage of the default learning rate",
        )

        parser.add_argument(
            "--smallest_lr_pct_lbl_attn",
            default=0.0001,
            type=float,
            help="Smallest non-encoder learning rate being a percentage of the default learning rate",
        )

        parser.add_argument(
            "--smallest_lr_pct_nonencoder",
            default=0.4,
            type=float,
            help="Smallest non-encoder learning rate being a percentage of the default learning rate",
        )

        parser.add_argument(
            "--n_lbl_attn_layer",
            default=1,
            type=int,
            help="Number of label attention layers.",
        )

        parser.add_argument(
            "--static_desc_emb",
            type=str2bool,
            default=True,
            help="Whether to update description embedding using BERT"
        )

        parser.add_argument(
            "--label_attn_lr",
            type=float,
            default=1e-04,
            help="Learning rate for label attention layers"
        )

        return parser

if __name__ == "__main__":

    seed_everything(69)

    hparams = Namespace(
        encoder_model="bert-base-cased",
        data_path="./project/data",
        dataset="hoc",
        batch_size=2,
        num_workers=2,
        random_sampling=False,
        num_frozen_epochs=1,
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
        hparams.tgt_txt_col, hparams.tgt_lbl_col,
    )

    num_classes = datamodule.num_classes

    model = HOCClassifier(
        hparams, tokenizer, collator, hparams.encoder_model,
        hparams.batch_size, num_classes, hparams.num_frozen_epochs,
        hparams.encoder_learning_rate, hparams.learning_rate,
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule)
