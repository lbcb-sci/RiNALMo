from lightning import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer

import re
import yaml

def _is_parent_module_unfrozen(module_name, potential_parent_modules):
    for potential_parent_module_name in potential_parent_modules.keys():
        if module_name.startswith(potential_parent_module_name):
            return True

    return False

class GradualUnfreezing(BaseFinetuning):
    def __init__(self, unfreeze_schedule_path: str):
        super().__init__()

        # Load unfreezing/fine-tuning schedule
        with open(unfreeze_schedule_path, "r") as f:
            self.unfreeze_schedule = yaml.safe_load(f)

        # "Merge" regexes for each epoch
        for epoch in self.unfreeze_schedule:
            self.unfreeze_schedule[epoch] = re.compile('|'.join(self.unfreeze_schedule[epoch]))

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        models_to_freeze = []
        for module_name, module in pl_module.named_modules():
            # Ignore root module (module_name = '')
            if not module_name:
                continue

            # Collect all modules that are not tuned in the first epoch
            if not bool(self.unfreeze_schedule[0].match(module_name)):
                models_to_freeze.append(module)

        # Freeze collected modules
        self.freeze(models_to_freeze)

    def finetune_function(self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer) -> None:
        if current_epoch in self.unfreeze_schedule and current_epoch != 0:
            modules_to_unfreeze = {}

            # Collect next phase modules
            for module_name, module in pl_module.named_modules():
                if bool(self.unfreeze_schedule[current_epoch].match(module_name)) and not _is_parent_module_unfrozen(module_name, modules_to_unfreeze):
                    modules_to_unfreeze[module_name] = module

            # Unfreeze collected modules
            self.unfreeze_and_add_param_group(
                modules=modules_to_unfreeze.values(),
                optimizer=optimizer,
                initial_denom_lr=1.0,
            )
