#!/usr/bin/env python
import argparse, json, time, torch, torchio as tio

from pathlib import Path
from torch.amp import autocast
from torch.utils.data import Dataset

from datasets import *
from models import Encoder


def get_args():
    p = argparse.ArgumentParser(
        description="Extract 128-d embedding from a single DTI tensor volume"
    )
    p.add_argument("--ckpt", required=True, help="Path to trained .pth checkpoint")
    p.add_argument("--tensor", required=True, help="Path to 4/6-ch tensor NIfTI/MHA")
    p.add_argument("--output", required=True, help="Path to save 128-d JSON vector")
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Force device (default: auto cuda if available)",
    )
    return p.parse_args()


def load_tensor(path: Path, dataset: Dataset, device: torch.device) -> torch.Tensor:
    VAL_TRANSFORM = dataset.transform
    subj = tio.Subject(dti=tio.ScalarImage(path))
    data = VAL_TRANSFORM(subj)["dti"].data.float()
    return data.unsqueeze(0).to(device, non_blocking=True)


@torch.inference_mode()
def forward_once(model: Encoder, tensor: torch.Tensor) -> torch.Tensor:
    with autocast(tensor.device.type):
        emb = model(tensor)
    return emb.squeeze(0)


def main():
    args = get_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    model = Encoder(layers=3).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    dataset = EigenvalueDataset(path_list=[], mode="val")
    tensor = load_tensor(Path(args.tensor), dataset, device)

    t0 = time.time()
    emb = forward_once(model, tensor)
    print(f"Inference done in {time.time()-t0:.2f}s")

    emb_list = emb.cpu().tolist()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(emb_list, f)
    print(f"Saved 128-d embedding to {args.output}")


if __name__ == "__main__":
    main()
