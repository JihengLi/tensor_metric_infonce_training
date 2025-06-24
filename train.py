# nohup python3.10/bin/python3.10 train.py > out.log 2>&1 &

import os, math, re
import torch

from datasets import *
from models import Encoder
from utils import (
    init_weights,
    process_train_batch,
    process_val_batch,
    save_checkpoint,
    gpu_safe_context,
)
from losses import nt_xent_loss

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler
from torch.utils.data import DataLoader

EPOCH_NUM = 12
NTX_LOSS_TEM = 0.3

if __name__ == "__main__":
    home_dir = "/home-local/lij112/codes/beyond_fa_challenge/beyond_fa_infonce/tensor_metric_infonce_training"

    input_file = f"{home_dir}/tensor_paths.txt"
    with open(input_file, "r") as f:
        final_paths = [Path(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(final_paths)} paths from {input_file}")

    train_paths, val_paths = train_test_split(
        final_paths, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(path_list=train_paths, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        # persistent_workers=True,
        # prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = TensorDataset(path_list=val_paths, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        # persistent_workers=True,
        # prefetch_factor=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(num_channels=6).to(device)
    model.apply(init_weights)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if n.endswith("bias") or "bn" in n else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-2},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-4,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        div_factor=10,
        final_div_factor=1e4,
        pct_start=0.1,
        epochs=EPOCH_NUM,
        steps_per_epoch=len(train_loader),
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    scaler = GradScaler(init_scale=2**14, growth_interval=2000)

    start_epoch = 1
    best_val_loss = math.inf
    ckpt_dir = f"{home_dir}/checkpoints"

    latest_epoch = -1
    pat = re.compile(r"dti_epoch(\d+)\.pth$")
    resume_path = None

    for fname in os.listdir(ckpt_dir):
        m = pat.match(fname)
        if m:
            ep = int(m.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                resume_path = os.path.join(ckpt_dir, fname)

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]

        print(
            f"Auto-resumed from {resume_path} — "
            f"epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})"
        )
    else:
        print("No checkpoint found — starting fresh training.")

    for epoch in range(start_epoch, EPOCH_NUM + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_step = 0
        with gpu_safe_context():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            for batch in pbar:
                loss = process_train_batch(
                    batch,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    lambda q, d: nt_xent_loss(q, d, NTX_LOSS_TEM),
                    scaler,
                    clip_grad=5.0,
                )
                if math.isnan(loss):
                    pbar.write("Skipping NaN batch")
                    continue
                train_step += 1
                train_loss += loss
                avg_loss = train_loss / train_step
                lr = scheduler.get_last_lr()[0] if scheduler else 0
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "avg": f"{avg_loss:.4f}",
                        "lr": f"{lr:.6f}",
                    }
                )
        avg_train = train_loss / train_step
        print(f"Epoch {epoch}/{EPOCH_NUM} - Train Loss: {avg_train:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_step = 0
        with gpu_safe_context():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                loss = process_val_batch(
                    batch,
                    model,
                    device,
                    lambda q, d: nt_xent_loss(q, d, NTX_LOSS_TEM),
                )
                if math.isnan(loss):
                    pbar.write("Skipping NaN batch")
                    continue
                val_step += 1
                val_loss += loss
                avg_loss = val_loss / val_step
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "avg": f"{avg_loss:.4f}",
                    }
                )
        avg_val = val_loss / val_step
        print(f"Epoch {epoch}/{EPOCH_NUM} - Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(
                f"{home_dir}/checkpoints/dti_best.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                scaler,
            )
            print(f"Best model updated at epoch {epoch}\nval_loss={best_val_loss:.4f}")

        save_checkpoint(
            f"{home_dir}/checkpoints/dti_epoch{epoch}.pth",
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
            scaler,
        )
