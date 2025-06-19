import gc
import torch
from torch.amp import autocast, GradScaler
from contextlib import contextmanager

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
    device,
    loss_fn,
    scaler: GradScaler,
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


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss, scaler):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "best_val_loss": best_val_loss,
        },
        path,
    )