import os
import torch
import torchio as tio
from torch.utils.data import Dataset
from torch.nn.functional import interpolate


class EigenvalueDataset(Dataset):
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
                        tio.Lambda(self._tensor_to_eigenvalues),
                        tio.Lambda(self._znorm_nonzero),
                        tio.RandomFlip(axes=(0, 1, 2), p=0.2),
                        tio.RandomAffine(
                            scales=(0.9, 1.1), degrees=15, translation=(5, 5, 5), p=0.2
                        ),
                        tio.Lambda(self._resize_to_64),
                        tio.RandomElasticDeformation(
                            num_control_points=9, max_displacement=1.5, p=0.2
                        ),
                        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
                        tio.RandomNoise(mean=0.0, std=(0.0, 0.25), p=0.3),
                        tio.RandomBlur(std=(0.5, 1.5), p=0.3),
                        tio.Lambda(self._clean_tensor),
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._jitter_background),
                    ]
                )
            else:
                self.transform = tio.Compose(
                    [
                        tio.Lambda(self._tensor_to_eigenvalues),
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._resize_to_64),
                    ]
                )

    def _tensor_to_eigenvalues(self, t: torch.Tensor) -> torch.Tensor:
        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = t
        A = torch.stack(
            [
                torch.stack([Dxx, Dxy, Dxz], 0),
                torch.stack([Dxy, Dyy, Dyz], 0),
                torch.stack([Dxz, Dyz, Dzz], 0),
            ]
        )  # shape (3, 3, D, H, W)
        A = A.permute(2, 3, 4, 0, 1).reshape(-1, 3, 3)  # (N, 3, 3)
        w = torch.linalg.eigvalsh(A).real  # (N, 3)
        w = w.flip(-1)
        w = w.reshape(t.shape[1], t.shape[2], t.shape[3], 3).permute(3, 0, 1, 2)
        return w.float()

    def _jitter_background(self, t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        mask = t == 0
        noise = torch.randn_like(t) * std
        return torch.where(mask, noise, t)

    def _znorm_nonzero(self, t: torch.Tensor) -> torch.Tensor:
        mask = t != 0
        if mask.any():
            vals = t[mask]
            mu, std = vals.mean(), vals.std()
            if std > 0:
                t = torch.where(mask, (t - mu) / std, t)
        return t

    def _resize_to_64(self, t: torch.Tensor) -> torch.Tensor:
        out = interpolate(
            t.unsqueeze(0),
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # (C, 64, 64, 64)
        return out

    def _clean_tensor(self, t: torch.Tensor) -> torch.Tensor:
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
