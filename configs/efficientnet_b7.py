import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from models import *
from datasets import *
from utils import *


class EfficientNetb7Config:
    def __init__(self, train_paths, val_paths, epochs, device):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.device = device
        self.epochs = epochs

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
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
        model = EfficientNetEncoder.from_name("efficientnet-b7", in_channels=6).to(
            self.device
        )
        model.apply(kaiming_normal_init)
        return model

    def _build_optimizer(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (no_decay if n.endswith("bias") or "bn" in n else decay).append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-2},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=5e-4,
            betas=(0.9, 0.95),
        )
        return optimizer

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            div_factor=5,
            final_div_factor=1e4,
            pct_start=0.1,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    def _build_scaler(self):
        return GradScaler(init_scale=2**14, growth_interval=2000)
