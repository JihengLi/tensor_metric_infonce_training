import gc, os, csv
import torch

from torch.amp import autocast, GradScaler
from contextlib import contextmanager
from tqdm import tqdm


@contextmanager
def gpu_safe_context():
    try:
        yield
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise e


def process_train_batch(
    batch,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    loss_fn,
    clip_grad: float | None = 5.0,
):
    optimizer.zero_grad(set_to_none=True)
    x1, x2 = (t.to(device, non_blocking=True) for t in batch)
    with autocast(device.type):
        z1, z2 = model(x1), model(x2)
        loss = loss_fn(z1, z2)
    scaler.scale(loss).backward()
    if clip_grad is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.detach().item()


@torch.inference_mode()
def process_val_batch(batch, model, device, loss_fn):
    x1, x2 = (t.to(device, non_blocking=True) for t in batch)
    with autocast(device.type):
        z1, z2 = model(x1), model(x2)
        loss = loss_fn(z1, z2)
    return loss.item()


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs,
    ckpt_dir,
    loss_fn,
    clip_grad: float | None = 5.0,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_cls",
                "train_reg",
                "val_loss",
                "val_cls",
                "val_reg",
                "lr",
            ]
        )
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        # ——— Training ———
        t_loss = 0.0
        steps = 0
        with gpu_safe_context():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            for batch in pbar:
                loss = process_train_batch(
                    batch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    loss_fn,
                    clip_grad,
                )
                steps += 1
                t_loss += loss
                t_avg = t_loss / steps
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "avg": f"{t_avg:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
        train_loss = t_loss / steps
        print(f"Epoch {epoch} — Train Avg Loss: {train_loss:.4f}")

        # ——— Validation ———
        v_loss = 0.0
        v_steps = 0
        with gpu_safe_context():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                loss = process_val_batch(batch, model, device, loss_fn)
                v_steps += 1
                v_loss += loss
                v_avg = v_loss / v_steps
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "v_avg": f"{v_avg:.4f}",
                    }
                )
        val_loss = v_loss / v_steps
        print(f"Epoch {epoch} — Val Avg Loss: {val_loss:.4f}")

        lr = scheduler.get_last_lr()[0]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.6e}",
                ]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                f"{ckpt_dir}/best.pth",
            )
            print(f">>> New best model at epoch {epoch}: {best_val_loss:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            f"{ckpt_dir}/epoch{epoch}.pth",
        )
    print("Training complete.")
