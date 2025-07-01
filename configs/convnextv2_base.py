import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from models import *
from datasets import *
from utils import *


class ConvNeXtV2BaseConfig:
    def __init__(self, train_paths, val_paths, epochs, device):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.device = device
        self.epochs = epochs

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler_cosannealing()
        self.scaler = self._build_scaler()

    def _build_dataloaders(self):
        train_dataset = TensorDataset(path_list=self.train_paths, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = TensorDataset(path_list=self.val_paths, mode="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _build_model(self):
        model = ConvNeXtV2Encoder(
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], proj_hidden_dim=2048
        ).to(self.device)
        return model

    def _build_optimizer(self):
        base_lr: float = 1e-4
        weight_decay: float = 1e-3

        visited = set()
        decay, no_decay = [], []

        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        blacklist_modules = (
            nn.LayerNorm,
            LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            GRN,
        )

        for module in self.model.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or param in visited:
                    continue
                visited.add(param)
                if isinstance(module, blacklist_modules) or param_name.endswith("bias"):
                    no_decay.append(param)
                elif isinstance(module, whitelist_weight_modules):
                    decay.append(param)
                else:
                    decay.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=base_lr,
            betas=(0.9, 0.999),
        )

    def _build_scheduler_cosannealing(
        self,
    ):
        warmup_pct: float = 0.10
        start_factor: float = 1e-3
        min_lr_ratio: float = 1 / 100

        steps_per_epoch = len(self.train_loader)
        total_steps = self.epochs * steps_per_epoch
        warm_steps = max(1, int(total_steps * warmup_pct))

        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warm_steps,
        )

        base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        eta_mins = [lr * min_lr_ratio for lr in base_lrs]

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warm_steps,
            eta_min=min(eta_mins),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warm_steps],
            last_epoch=-1,
        )

    def _build_scheduler_onecycle(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=5e-4,
            div_factor=5,
            final_div_factor=1e2,
            pct_start=0.1,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    def _build_scaler(self):
        return GradScaler(init_scale=2**14, growth_interval=2000)
