# nohup python3.10/bin/python3.10 train.py > out.log 2>&1 &

import os, math, re
import torch

from datasets import *
from models import *
from utils import *
from losses import *
from configs import *

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ConvNeXtV2BaseConfig(
        train_paths=train_paths, val_paths=val_paths, epochs=EPOCH_NUM, device=device
    )
    train_loader = cfg.train_loader
    val_loader = cfg.val_loader
    model = cfg.model
    optimizer = cfg.optimizer
    scheduler = cfg.scheduler
    scaler = cfg.scaler

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
