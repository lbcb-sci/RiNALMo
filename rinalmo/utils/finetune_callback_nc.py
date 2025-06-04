from lightning import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer

import re
import yaml

def _collect_leaf_module_names(pl_module: LightningModule):
    leaf_module_names = set()

    for module_name, _ in pl_module.named_modules():
        # Ignore root module (module_name = '')
        if not module_name:
            continue

        is_module_redundant = False

        for leaf_module_name in set(leaf_module_names):
            module_parent_name = '.'.join(module_name.split('.')[:-1])
            if module_parent_name == leaf_module_name:
                # Parent found, replace it with the child module
                leaf_module_names.remove(leaf_module_name)
                break
            elif leaf_module_name == module_name:
                # Child found, currently considered module is redundant
                is_module_redundant = True
                break

        if not is_module_redundant:
            leaf_module_names.add(module_name)

    return leaf_module_names

class GradualUnfreezing(BaseFinetuning):
    def __init__(self, unfreeze_schedule_path: str):
        super().__init__()

        # Load unfreezing/fine-tuning schedule
        with open(unfreeze_schedule_path, "r") as f:
            self.unfreeze_schedule = yaml.safe_load(f)

        # "Merge" regexes for each epoch
        for epoch in self.unfreeze_schedule:
            self.unfreeze_schedule[epoch] = re.compile('|'.join(self.unfreeze_schedule[epoch]))

        self.leaf_module_names = None

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        modules_to_freeze = []
        self.leaf_module_names = _collect_leaf_module_names(pl_module)

        for module_name, module in pl_module.named_modules():
            # Skip non-leaf modules
            if module_name not in self.leaf_module_names:
                continue

            # Collect all modules that are not tuned in the first (0-th) epoch
            if not self.unfreeze_schedule[0].match(module_name):
                modules_to_freeze.append(module)

        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                # freeze beta params from SwiGLU
                if name.find('beta') != -1:
                    param.requires_grad = False

        # Freeze collected modules
        self.freeze(modules_to_freeze)

    def finetune_function(self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer) -> None:
        if current_epoch in self.unfreeze_schedule and current_epoch != 0:
            modules_to_unfreeze = []

            for module_name, module in pl_module.named_modules():
                # Skip non-leaf modules
                if module_name not in self.leaf_module_names:
                    continue

                # Collect modules to unfreeze
                if self.unfreeze_schedule[current_epoch].match(module_name):
                    modules_to_unfreeze.append(module)
                    # unfreeze beta params from SwiGLU
                    if module_name.find('linear_gate') != -1:
                        for param_name, param in pl_module.named_parameters():
                            if param_name == '.'.join(module_name.split('.')[:-1]) + '.beta':
                                param.requires_grad = True

            # Unfreeze collected modules
            self.unfreeze_and_add_param_group(
                modules=modules_to_unfreeze,
                optimizer=optimizer,
                initial_denom_lr=1.0,
            )