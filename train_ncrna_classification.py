import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import lightning.pytorch as pl

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.ncrna_classification.datamodule import ncRNADataModule
from rinalmo.model.model import RiNALMo
from rinalmo.utils.finetune_callback import GradualUnfreezing
from rinalmo.model.downstream import ncRNAClassificationHead
from rinalmo.config import model_config

import argparse
from pathlib import Path
from datetime import timedelta

from sklearn.metrics import f1_score, accuracy_score

class ncRNAClassificationWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        head_embed_dim: int = 256,
        n_classes: int = 88,
        lr: float = 8e-6,
        weight_decay: float = 0.01,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        self.rinalmo = RiNALMo(model_config(lm_config))
        self.pred_head = ncRNAClassificationHead(
            c_in=self.rinalmo.config['model']['transformer'].embed_dim,
            embed_dim=head_embed_dim,
            n_classes=n_classes,
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        self.val_step_outputs = []

    def load_pretrained_rinalmo_weights(self, pretrained_weights_path):
        self.rinalmo.load_state_dict(torch.load(pretrained_weights_path))

    def forward(self, tokens):
        x = self.rinalmo(tokens)["representation"]
        x = x[:, 0] # take the CLS token, B x E
            
        pred = self.pred_head(x) # B x C
        return pred

    def _common_step(self, batch, batch_idx, split):
        fam, tokens, labels = batch
        pred = self(tokens)

        loss = self.loss_fn(pred, labels)

        log = {
            f'{split}/loss': loss,
        }

        self.log_dict(log, sync_dist=True, add_dataloader_idx=False)

        return loss
    
    def _validation_step(self, batch, batch_idx, split: str):
        fam, tokens, labels = batch
        pred = self(tokens)

        loss = self.loss_fn(pred, labels)

        self.val_step_outputs.append((torch.argmax(pred, dim=-1), labels))

        log = {
            f'{split}/loss': loss,
        }

        self.log_dict(log, sync_dist=True, add_dataloader_idx=False)

        return loss

    def _on_epoch_end(self, split: str):
        preds = []
        labels = []

        for pred, label in self.val_step_outputs:
            preds.append(pred)
            labels.append(label)
        
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # calculate F1 score
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)

        log = {
            f'{split}/f1': f1,
            f'{split}/acc': acc,
        }

        self.log_dict(log, sync_dist=True, add_dataloader_idx=False)

        self.val_step_outputs.clear() 
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx, split='val')
    
    def on_validation_epoch_end(self):
        self._on_epoch_end(split='val')
    
    def test_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx, split='test')
    
    def on_test_epoch_end(self):
        self._on_epoch_end(split='test')

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=7500)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }
    
def main(args):
    if args.seed:
        pl.seed_everything(args.seed)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Model
    model = ncRNAClassificationWrapper(
        lm_config=args.lm_config,
        head_embed_dim=args.head_embed_dim,
        n_classes=args.n_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.pretrained_rinalmo_weights:
        model.load_pretrained_rinalmo_weights(args.pretrained_rinalmo_weights)

    if args.init_params:
        # check if init_params ends with .ckpt
        if args.init_params.endswith('.ckpt'):
            model = ncRNAClassificationWrapper.load_from_checkpoint(args.init_params)
        else:
            model.load_state_dict(torch.load(args.init_params))

    # Data
    alphabet = Alphabet()
    datamodule = ncRNADataModule(
        data_root=args.data_dir,
        boundary_noise=args.boundary_noise,
        alphabet=alphabet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        skip_data_preparation=not args.prepare_data,
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
            filename='ncRNA_classification-epoch_ckpt-{epoch}-{step}',
            every_n_epochs=1,
            save_top_k=-1
        )
        callbacks.append(epoch_ckpt_callback)

    if args.checkpoint_every_epoch_top_1:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='ncRNA_classification-epoch_ckpt-{epoch}-{step}',
            every_n_epochs=1,
            save_top_k=1,
            monitor='val/acc',
            mode='max'
        )
        callbacks.append(epoch_ckpt_callback)

    if args.ft_schedule:
        ft_callback = GradualUnfreezing(
            unfreeze_schedule_path=args.ft_schedule,
        )
        callbacks.append(ft_callback)

    # Training
    strategy='auto'
    if args.devices != 'auto' and ("," in args.devices or (int(args.devices) > 1 and int(args.devices) != -1)):
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_every_n_steps,
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
    )

    if not args.test_only:
        trainer.fit(model=model, datamodule=datamodule)
        model = ncRNAClassificationWrapper.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "data_dir", type=str, 
        help="Path to the directory containing the data files."
    )
    # Optional
    parser.add_argument(
        "--prepare_data", action="store_true", default=False,
        help="Download and prepare training, validation and test data"
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
        "--checkpoint_every_epoch_top_1", action="store_true", default=False,
        help="""Whether to checkpoint at the end of every training epoch
              based on the top-1 validation accuracy"""
    )
    parser.add_argument(
        "--test_only", action="store_true", default=False,
        help="""
        Skip the training and only run the evaluation on the test set
        (make sure to set '--init_params' if you are using this option)
        """
    )
    parser.add_argument(
        "--init_params", type=str, default=None,
        help="""
        Path to the '.pt' file containing model weights that will be used
        as the starting point for the training (or evaluation)
        """
    )

    # Data
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="How many subprocesses to use for data loading"
    )
    parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help=" If activated, the data loader will copy Tensors into device/CUDA pinned memory before returning them"
    )

    # Model
    parser.add_argument(
        "--lm_config", type=str, default="giga",
        help="Language model configuration"
    )
    parser.add_argument(
        "--pretrained_rinalmo_weights", type=str, default=None,
        help="Path to the pretrained RiNALMo model weights"
    )
    parser.add_argument(
        "--head_embed_dim", type=int, default=256,
        help="Embedding dimension of the classification head"
    )
    parser.add_argument(
        "--n_classes", type=int, default=88,
        help="Number of classes"
    )
    parser.add_argument(
        "--boundary_noise", type=str, default='',
        help="Boundary noise for the model ('' or '_bn200')"
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

    # Training
    parser.add_argument(
        "--ft_schedule", type=str, default=None,
        help="Path to the fine-tuning schedule file"
    )
    parser.add_argument(
        "--lr", type=float, default=8e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay"
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
        "--precision", type=str, default='16-mixed',
        help="Double precision, full precision, 16bit mixed precision or bfloat16 mixed precision"
    )

    args = parser.parse_args()
    main(args)