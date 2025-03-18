import torch
import torch.nn as nn
from torch.optim import AdamW

import pytorch_lightning as pl

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from torchmetrics.functional.classification import binary_confusion_matrix

import argparse
from pathlib import Path
from datetime import timedelta

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.splice_site_prediction.datamodule import SpliceSiteDataModule
from rinalmo.model.model import RiNALMo
from rinalmo.model.downstream import SpliceSitePredictionHead
from rinalmo.config import model_config
import rinalmo.utils.splice_site_metrics as ss_pred_metrics

PRED_HEAD_EMBED_DIM = 128
class SpliceSitePredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        head_embed_dim: int = 128,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()

        self.rinalmo = RiNALMo(model_config(lm_config))
        self.pred_head = SpliceSitePredictionHead(
            c_in=self.rinalmo.config['model']['transformer'].embed_dim,
            embed_dim=head_embed_dim,
        )

        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        self.val_step_outputs = []
    
    def load_pretrained_rinalmo_weights(self, pretrained_weights_path):
        self.rinalmo.load_state_dict(torch.load(pretrained_weights_path))

    def forward(self, tokens):

        x = self.rinalmo(tokens)["representation"]
        x = x[:, 0]
            
        pred = self.pred_head(x)
        return pred
    
    def _common_step(self, batch, batch_idx, log_prefix: str):
        seq, labels = batch
        labels = labels.unsqueeze(dim=1)
        preds = self(seq)

        loss = self.loss(preds, labels)

        log = {
            f'{log_prefix}/loss': loss,
        }
        self.log_dict(log, sync_dist=True, add_dataloader_idx=False)

        return loss
    
    def _validation_step(self, batch, batch_idx, log_prefix: str):
        seq, labels = batch
        labels = labels.unsqueeze(dim=1)
        preds = self(seq)

        loss = self.loss(preds, labels)

        matrix = binary_confusion_matrix(preds, labels)
        self.val_step_outputs.append(matrix)

        log = {
            f'{log_prefix}/loss': loss,
        }
        self.log_dict(log, sync_dist=True)

        return loss, matrix

    def _on_epoch_end(self, log_prefix: str):
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for matrix in self.val_step_outputs:
            tn += matrix[0][0]
            tp += matrix[1][1]
            fp += matrix[0][1]
            fn += matrix[1][0]

        _all = tp + tn + fp + fn 
        acc = ss_pred_metrics.accuracy(tp, tn, _all)
        prec = ss_pred_metrics.precision(tp, fp) if ((tp > 0) or (fp > 0)) else 0
        recall = ss_pred_metrics.recall(tp, fn) if ((tp > 0) or (fn > 0)) else 0
        specificity = ss_pred_metrics.specificity(tn, fp) if ((tn > 0) or (fp > 0)) else 0
        f1 = ss_pred_metrics.f1_score(prec, recall) if ((prec > 0) or (recall > 0)) else 0

        log = {
            f'{log_prefix}/acc': acc,
            f'{log_prefix}/precision': prec,
            f'{log_prefix}/recall': recall,
            f'{log_prefix}/specificity': specificity,
            f'{log_prefix}/f1_score': f1,
        }

        self.log_dict(log, sync_dist=True, add_dataloader_idx=False)

        self.val_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, log_prefix="train")
    
    def validation_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx, log_prefix=f"val")
    
    def on_validation_epoch_end(self):
        self._on_epoch_end(log_prefix='val')

    def test_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx, log_prefix=f"test")

    def on_test_epoch_end(self):
        self._on_epoch_end(log_prefix='test')

    def configure_optimizers(self):
        optimizer = AdamW([{'params': self.pred_head.parameters()},
                           {'params': self.rinalmo.transformer.parameters()},
                          ], lr=self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }
    
def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Model
    model = SpliceSitePredictionWrapper(
        lm_config=args.lm_config,
        head_embed_dim=PRED_HEAD_EMBED_DIM,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.pretrained_rinalmo_weights:
        model.load_pretrained_rinalmo_weights(args.pretrained_rinalmo_weights)

    if args.init_params:
        model.load_state_dict(torch.load(args.init_params))

    # Datamodule
    alphabet = Alphabet()
    datamodule = SpliceSiteDataModule(
        ss_type=args.ss_type,
        species=args.benchmark,
        dataset_id=args.dataset_id,
        data_root=args.data_dir,
        test_data_root=args.test_data_dir,
        alphabet=alphabet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        skip_data_preparation = not args.prepare_data
    )

    # Set up callbacks and loggers
    callbacks = []
    loggers = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.wandb_experiment_name,
            save_dir=args.output_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_code=True,
        )
        loggers.append(wandb_logger)

    if args.log_lr and loggers:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    if args.checkpoint_every_epoch:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='ss_pred-epoch_ckpt-{epoch}-{step}',
            every_n_epochs=1,
            save_top_k=-1
        )
        callbacks.append(epoch_ckpt_callback)

    if args.checkpoint_every_hour:
        time_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='ss_pred-latest-hourly-{epoch}-{step}',
            train_time_interval=timedelta(hours=1.0),
            save_top_k=1
        )
        callbacks.append(time_ckpt_callback)

    # Training
    strategy='auto'
    if args.devices != 'auto' and ("," in args.devices or (int(args.devices) > 1 and int(args.devices) != -1)):
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_every_n_steps,
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
    )

    if args.data_dir:
        trainer.fit(model=model, datamodule=datamodule)
    if args.test_data_dir:
        trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lm_config", type=str, default="giga",
        help="Language model configuration"
    )
    parser.add_argument(
        "--pretrained_rinalmo_weights", type=str, default=None,
        help="Path to the pretrained RiNALMo model weights"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory with all the training and evaluation data"
    )
    parser.add_argument(
        "--prepare_data", action="store_true", default=False,
        help="Download and prepare training, validation and test data"
    )
    parser.add_argument(
        "--test_data_dir", type=str, default=None,
        help="Directory with all the test data"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for all the output files (checkpoints, logs, temporary files, etc.)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="Whether to checkpoint at the end of every training epoch"
    )
    parser.add_argument(
        "--checkpoint_every_hour", action="store_true", default=False,
        help="Whether to checkpoint every hour during the training (each checkpoint overwrites the last one)"
    )
    parser.add_argument(
        "--init_params", type=str, default=None,
        help="""
        Path to the '.pt' file containing model weights that will be used
        as the starting point for the training (or evaluation)
        """
    )

    # W&B
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="Wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50,
        help="How often to log within steps"
    )

    # Data
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="How many samples per batch to load"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="How many subprocesses to use for data loading"
    )
    parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help=" If activated, the data loader will copy Tensors into device/CUDA pinned memory before returning them"
    )

    # Training
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6,
        help="Weight decay coefficient"
    )
    parser.add_argument(
        "--accelerator", type=str, default='auto',
        help="Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps”, “auto”)"
    )
    parser.add_argument(
        "--devices", type=str, default='auto',
        help="The devices to use for training"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Stop training after this number of steps"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=-1,
        help=" Stop training once this number of epochs is reached"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=None,
        help="The value at which to clip gradients"
    )
    parser.add_argument(
        "--precision", type=str, default='bf16-mixed',
        help="Double precision, full precision, 16bit mixed precision or bfloat16 mixed precision"
    )

    # Prediction type
    parser.add_argument(
        "--ss_type", type=str, default='donor',
        help="Whether donor or acceptor SS prediction type"
    )
    parser.add_argument(
        "--dataset_id", type=str, default='db_1',
        help="Dataset, {db_1 .. db_10}"
    )
    # Benchmark dataset
    parser.add_argument(
        "--benchmark", type=str, default='Danio',
        help="Benchmark dataset: (“Danio”, “Fly”, “Thaliana”, “Worm”)"
    )

    args = parser.parse_args()
    main(args)