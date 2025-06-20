import os
import torch
import torchio as tio
from torch.utils.data import Dataset


def _tensor_to_eigenvalues(t: torch.Tensor) -> torch.Tensor:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = t
    A = torch.stack(
        [
            torch.stack([Dxx, Dxy, Dxz], 0),
            torch.stack([Dxy, Dyy, Dyz], 0),
            torch.stack([Dxz, Dyz, Dzz], 0),
        ]
    )  # shape (3, 3, D, H, W)
    A = A.permute(2, 3, 4, 0, 1).reshape(-1, 3, 3)  # (N, 3, 3)
    w = torch.linalg.eigvalsh(A).real  # (N, 3), already λ₁≤λ₂≤λ₃
    w = w.flip(-1)  # λ₁≥λ₂≥λ₃
    w = w.reshape(t.shape[1], t.shape[2], t.shape[3], 3).permute(3, 0, 1, 2)
    return w.float()


class EigenvalueDataset(Dataset):
    def __init__(self, path_list: list[str], transform: tio.Transform | None = None):
        self.sample_paths = path_list
        self.transform = transform or tio.Compose(
            [
                tio.Lambda(lambda x: _tensor_to_eigenvalues(x)),
                tio.Lambda(self._clean),
                tio.ZNormalization(),
                tio.RandomFlip(axes=(0, 1, 2), p=0.5),
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=15,
                    translation=(5, 5, 5),
                    p=0.7,
                ),
                tio.RandomElasticDeformation(
                    num_control_points=7, max_displacement=7.0, p=0.2
                ),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
                tio.RandomNoise(mean=0.0, std=(0.0, 0.25), p=0.3),
                tio.RandomBlur(std=(0.5, 1.5), p=0.2),
                tio.CropOrPad((64, 64, 64)),
            ]
        )

    @staticmethod
    def _clean(t: torch.Tensor):
        t = t.clone()
        t[torch.isnan(t) | torch.isinf(t)] = 0
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
                subject = tio.Subject(dti=tio.ScalarImage(nii_path))
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
