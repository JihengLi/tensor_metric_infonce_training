import os
import torch
import torchio as tio
from torch.utils.data import Dataset
from torch.nn.functional import interpolate


class TensorDataset(Dataset):
    """Dataset for Diffusion Tensor Imaging contrastive learning."""

    def __init__(
        self,
        path_list: list[str],
        mode: str = "train",
        transform: tio.Transform | None = None,
    ):
        self.sample_paths = path_list

        if transform is not None:
            self.transform = transform
        else:
            if mode == "train":
                self.transform = tio.Compose(
                    [
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._resize_to_64),
                        tio.RandomGamma(log_gamma=(-0.3, 0.3)),  # contrast/brightness
                        tio.RandomNoise(mean=0.0, std=(0, 0.25)),
                        tio.RandomBlur(std=(0.5, 1.5)),
                        tio.Lambda(self._clean_tensor),
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._jitter_background),
                    ]
                )
            else:
                self.transform = tio.Compose(
                    [
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._resize_to_64),
                    ]
                )

    @staticmethod
    def _jitter_background(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        mask = t == 0
        noise = torch.randn_like(t) * std
        return torch.where(mask, noise, t)

    @staticmethod
    def _znorm_nonzero(t: torch.Tensor) -> torch.Tensor:
        mask = t != 0
        if mask.any():
            vals = t[mask]
            mu, std = vals.mean(), vals.std()
            if std > 0:
                t = torch.where(mask, (t - mu) / std, t)
        return t

    @staticmethod
    def _resize_to_64(t: torch.Tensor) -> torch.Tensor:
        out = interpolate(
            t.unsqueeze(0),
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # (C, 64, 64, 64)
        return out

    @staticmethod
    def _clean_tensor(t: torch.Tensor) -> torch.Tensor:
        t = torch.nan_to_num(t, nan=0.0)
        finite = t[torch.isfinite(t)]
        if finite.numel() > 0:
            mx, mn = finite.max(), finite.min()
            t = torch.where(t == float("inf"), mx, t)
            t = torch.where(t == float("-inf"), mn, t)
        return t

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        max_attempts = 5
        for _ in range(max_attempts):
            nii_path = self.sample_paths[idx]
            if not os.path.isfile(nii_path):
                print(f"[Warning] File not found: {nii_path}, trying next index...")
                idx = (idx + 1) % len(self.sample_paths)
                continue
            try:
                img = tio.ScalarImage(nii_path).data.float()
                img = self._clean_tensor(img)
                if (img == 0).all() or (img.max() - img.min()) < 1e-6:
                    idx = (idx + 1) % len(self.sample_paths)
                    continue
                subject = tio.Subject(dti=tio.ScalarImage(tensor=img))
                v1 = self.transform(subject)["dti"].data.float()
                v2 = self.transform(subject)["dti"].data.float()
                return v1, v2
            except Exception as e:
                print(
                    f"[Warning] Failed to process {nii_path}: {e}, trying next index..."
                )
                idx = (idx + 1) % len(self.sample_paths)
        raise RuntimeError(
            f"[Error] Failed to load a valid sample after {max_attempts} attempts."
        )
