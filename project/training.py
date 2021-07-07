"""
Runs a model on a single node across N-gpus.
"""
import argparse
import os
from datetime import datetime

from base_classifier import BaseClassifier
from hoc import HOCClassifier
from mtc import MTCClassifier 
from datamodule import MedDataModule, Collator
from tokenizer import Tokenizer
from utils import parse_dataset_name

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torchnlp.random import set_seed


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    seed_everything(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------

    tokenizer = Tokenizer(hparams.encoder_model)
    collator = Collator(tokenizer)
    datamodule = MedDataModule(
        tokenizer, collator, hparams.data_path,
        hparams.dataset, hparams.batch_size, 
        hparams.num_workers,
    )
    
    if hparams.dataset == 'hoc':
        model = HOCClassifier(
            hparams, tokenizer, collator, hparams.encoder_model,
            hparams.batch_size, hparams.nr_frozen_epochs,
            hparams.encoder_learning_rate, hparams.learning_rate,
        )
    else:
        model = MTCClassifier(
            hparams, tokenizer, collator, hparams.encoder_model,
            hparams.batch_size, hparams.nr_frozen_epochs,
            hparams.encoder_learning_rate, hparams.learning_rate,
        )

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=hparams.early_stop_min_delta,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    # ------------------------
    # 3 INIT LOGGERS
    # ------------------------
    # Tensorboard Callback
    exp_version = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    exp_name = exp_version + f"--{hparams.encoder_model}--{hparams.dataset}"
    
    tb_logger = TensorBoardLogger(
        save_dir=hparams.log_dir,
        version=exp_version,
        name=exp_name,
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        hparams.log_dir, exp_name, "checkpoints",
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        every_n_val_epochs=1,
        mode=hparams.metric_mode,
        save_weights_only=True
    )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[early_stop_callback,
                   checkpoint_callback],
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
        accelerator="ddp",
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule)
    
    trainer.test()


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=69, help="Training seed.")
    parser.add_argument(
        "--encoder_model",
        default="bert-base-uncased",
        type=str,
        help="Encoder model to be used.",
    )
    parser.add_argument(
        "--save_top_k",
        default=2,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str,
        help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=15,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=25,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    
    parser.add_argument(
        "--early_stop_min_delta",
        default=0.0,
        type=float,
        help=(
            "Delta for early stopping the model"
        ),
    )
    
    parser.add_argument(
        "--log_dir",
        default="lyta-experiments/",
        type=str,
        help=(
            "Logging of experiments and hparams directory."
        ),
    )

    # each LightningModule defines arguments relevant to it
    parser = MedDataModule.add_model_specific_args(parser)
    parser = BaseClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()
    
    hparams.dataset = parse_dataset_name(hparams.dataset)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
