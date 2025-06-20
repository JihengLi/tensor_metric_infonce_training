import os
import torch
import torchio as tio
from torch.utils.data import Dataset


class VectorRandomFlip(tio.RandomFlip):
    def apply_transform(self, subject):
        axis = self.axes if isinstance(self.axes, int) else self.axes[0]
        flipped = super().apply_transform(subject)
        v = flipped.dti.data[3:6]
        v[axis] = -v[axis]
        return flipped


class VectorRandomAffine(tio.RandomAffine):
    def apply_transform(self, subject):
        affine_params = self.get_parameters(random=self.is_random)
        R = torch.tensor(affine_params["matrix"][:3, :3], dtype=torch.float32)
        transformed = tio.transforms.Affine(
            matrix=affine_params["matrix"],
            image_interpolation=self.image_interpolation,
        )(subject)
        v = transformed.dti.data[3:6].reshape(3, -1)
        v = (R @ v).reshape(*transformed.dti.data[3:6].shape)
        transformed.dti.data[3:6] = v
        return transformed


def _tensor_to_eigvals_v1(t: torch.Tensor) -> torch.Tensor:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = t
    A = torch.stack(
        [
            torch.stack([Dxx, Dxy, Dxz], 0),
            torch.stack([Dxy, Dyy, Dyz], 0),
            torch.stack([Dxz, Dyz, Dzz], 0),
        ]
    )  # (3,3,D,H,W)
    A = A.permute(2, 3, 4, 0, 1).reshape(-1, 3, 3)  # (N,3,3)
    w, v = torch.linalg.eigh(A)
    w = w.flip(-1)
    v = v.flip(-1)
    v1 = v[..., 0]
    out = torch.cat([w, v1], dim=-1)  # (N,6)
    out = out.reshape(t.shape[1], t.shape[2], t.shape[3], 6).permute(
        3, 0, 1, 2
    )  # (6,D,H,W)
    return out.float()


class EigenvalueVectorDataset(Dataset):
    def __init__(self, path_list: list[str], transform: tio.Transform | None = None):
        self.sample_paths = path_list
        self.transform = transform or tio.Compose(
            [
                tio.Lambda(lambda x: _tensor_to_eigvals_v1(x)),
                tio.Lambda(self._clean),
                tio.ZNormalization(),
                VectorRandomFlip(axes=(0, 1, 2), p=0.5),
                VectorRandomAffine(
                    scales=(0.9, 1.1), degrees=15, translation=(5, 5, 5), p=0.7
                ),
                # TODO: VectorRandomElasticDeformation
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
