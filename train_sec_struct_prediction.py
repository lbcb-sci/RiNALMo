from typing import Mapping, Any
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import argparse
from collections import defaultdict
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.secondary_structure.datamodule import SecondaryStructureDataModule, SUPPORTED_DATASETS
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct, ss_precision, ss_recall, ss_f1, save_to_ct
from rinalmo.utils.finetune_callback import GradualUnfreezing
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config

THRESHOLD_TUNE_METRIC = "f1"
THRESHOLD_CANDIDATES = [i / 100 for i in range(1, 30, 1)]

class SecStructPredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        tune_threshold_every_n_epoch: int = 1,
        num_resnet_blocks: int = 2,
        lr: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lm = RiNALMo(model_config(lm_config))
        self.pred_head = SecStructPredictionHead(self.lm.config['model']['transformer'].embed_dim, num_blocks=num_resnet_blocks)

        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

        self._eval_step_outputs = None
        self.tune_threshold_every_n_epoch = tune_threshold_every_n_epoch

        self.threshold = 0.5

    def load_pretrained_lm_weights(self, pretrained_weights_path):
        self.lm.load_state_dict(torch.load(pretrained_weights_path))

    def forward(self, tokens):
        x = self.lm(tokens)["representation"]
        logits = self.pred_head(x[..., 1:-1, :]).squeeze(-1)

        return logits
    
    def _common_step(self, batch, batch_idx, log_prefix):
        *_, tokens, sec_struct_true = batch
        seq_len = sec_struct_true.shape[1]

        logits = self(tokens)
        upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits.device), diagonal=1)
        loss = self.loss(logits[..., upper_tri_mask], sec_struct_true[..., upper_tri_mask])

        self.log(f"{log_prefix}/loss", loss)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx, log_prefix="train")
        return loss

    def on_validation_start(self):
        self._reset_eval_step_outputs()

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, thresholds=THRESHOLD_CANDIDATES, log_prefix="val")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch != 0:
            # If sanity checking, don't update the threshold
            return

        # Find threshold with highest validation score
        best_metric_val = -1.0
        best_threshold = 0.0
        for threshold in THRESHOLD_CANDIDATES:
            curr_metric_val = sum(self._eval_step_outputs[THRESHOLD_TUNE_METRIC][threshold]) / len(self._eval_step_outputs[THRESHOLD_TUNE_METRIC][threshold])

            if curr_metric_val > best_metric_val:
                best_metric_val = curr_metric_val
                best_threshold = threshold

        self.threshold = best_threshold

        self.log_dict(
            {
                f"val/{THRESHOLD_TUNE_METRIC}": best_metric_val,
                f"val/threshold": self.threshold
            }
        )

    def on_test_start(self):
        self._reset_eval_step_outputs()

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, thresholds=[self.threshold], log_prefix="test")

    def on_test_epoch_end(self):
        # Get macro average of each metric
        for key in self._eval_step_outputs:
            metric_avg_val = sum(self._eval_step_outputs[key][self.threshold]) / len(self._eval_step_outputs[key][self.threshold])
            self.log(f"test/{key.lower()}", metric_avg_val)

    def _update_eval_step_outputs(self, logits, sec_struct_true, ss_ids, seqs, thresholds):
        batch_size, *_ = logits.shape

        probs = torch.sigmoid(logits)

        if probs.dtype == torch.bfloat16:
            # Cast brain floating point into floating point
            probs = probs.type(torch.float16)

        probs = probs.cpu().numpy()
        sec_struct_true = sec_struct_true.cpu().numpy()

        for i in range(batch_size):
            for threshold in thresholds:
                sec_struct_pred = prob_mat_to_sec_struct(probs=probs[i], seq=seqs[i], threshold=threshold)

                y_true = sec_struct_true[i]
                y_pred = sec_struct_pred

                self._eval_step_outputs["precision"][threshold].append(ss_precision(y_true, y_pred))
                self._eval_step_outputs["recall"][threshold].append(ss_recall(y_true, y_pred))
                self._eval_step_outputs["f1"][threshold].append(ss_f1(y_true, y_pred))

            if self.trainer.testing:
                output_dir = Path(self.trainer.default_root_dir)
                f1_score = self._eval_step_outputs["f1"][threshold][-1]

                save_to_ct(output_dir / f"{ss_ids[i]}_pred_f1={f1_score}.ct", sec_struct=y_pred, seq=seqs[i])

    def _reset_eval_step_outputs(self):
        self._eval_step_outputs = defaultdict(lambda: defaultdict(list))

    def _eval_step(self, batch, batch_idx, thresholds, log_prefix="eval"):
        ss_ids, seqs, _, sec_struct_true = batch
        _, logits = self._common_step(batch, batch_idx, log_prefix=log_prefix)

        if self.trainer.testing or (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch == 0:
            self._update_eval_step_outputs(logits, sec_struct_true, ss_ids, seqs, thresholds)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=7000) # TODO: Hardcoded

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        state_dict = dict(state_dict)
        self.threshold = state_dict["threshold"]
        state_dict.pop("threshold") # Remove 'threshold' key for possible "strict" clashes

        return super().load_state_dict(state_dict, strict, assign)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict']['threshold'] = self.threshold
        super().on_save_checkpoint(checkpoint)

def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Model
    model = SecStructPredictionWrapper(
        lm_config=args.lm_config,
        tune_threshold_every_n_epoch=args.tune_threshold_every_n_epoch,
        num_resnet_blocks=args.num_resnet_blocks,
        lr=args.lr
    )

    if args.pretrained_rinalmo_weights:
        model.load_pretrained_lm_weights(args.pretrained_rinalmo_weights)

    if args.init_params:
        model.load_state_dict(torch.load(args.init_params))

    # Data
    alphabet = Alphabet(**model.lm.config['alphabet'])

    datamodule = SecondaryStructureDataModule(
        data_root=args.data_dir,
        alphabet=alphabet,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        dataset=args.dataset,
        skip_data_preparation=not args.prepare_data,
    )

    # Callbacks
    callbacks = []

    if args.checkpoint_every_epoch:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='ss-epoch_ckpt-{epoch}-{step}',
            every_n_epochs=1,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        )
        callbacks.append(epoch_ckpt_callback)

    if args.ft_schedule:
        ft_callback = GradualUnfreezing(
            unfreeze_schedule_path=args.ft_schedule,
        )
        callbacks.append(ft_callback)

    # Loggers
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

    if loggers:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Training
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        default_root_dir=args.output_dir,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        logger=loggers,
        precision=args.precision,
    )

    if not args.test_only:
        trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=str,
        help="Directory containing training, validation and test data"
    )
    parser.add_argument(
        "--init_params", type=str, default=None,
        help="""
        Path to the '.pt' file containing model weights that will be used
        as the starting point for the training (or evaluation)
        """
    )
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
        "--tune_threshold_every_n_epoch", type=int, default=1,
        help="Tune positive class threshold after every N training epochs"
    )
    parser.add_argument(
        "--test_only", action="store_true", default=False,
        help="""
        Skip the training and only run the evaluation on the test set
        (make sure to set '--init_params' if you are using this option)
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
    parser.add_argument(
        "--min_seq_len", type=int, default=0,
        help="""
        All secondary structures for sequences that have less than this number of nucleotides
        will be discarded and won't be used during training and/or evaluation.
        """
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=999_999_999,
        help="""
        All secondary structures for sequences that have more than this number of nucleotides
        will be discarded and won't be used during training and/or evaluation.
        """
    )
    parser.add_argument(
        "--dataset", type=str, default="bpRNA",
        help=f"""
        Dataset that will be used for training, validation and testing. Options: {SUPPORTED_DATASETS}
        """
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
        "--num_resnet_blocks", type=int, default=2,
        help="Number of ResNet blocks used in secondary structure prediction head"
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
        "--lr", type=float, default=5e-4,
        help="Learning rate"
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
