import os
import torch
import torchio as tio
from torch.utils.data import Dataset
from torch.nn.functional import interpolate


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


class EigenvalueVectorDataset(Dataset):
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
                        tio.Lambda(self._tensor_to_eigvals_v1),
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._resize_to_64),
                        VectorRandomFlip(axes=(0, 1, 2), p=0.5),
                        VectorRandomAffine(
                            scales=(0.9, 1.1), degrees=15, translation=(5, 5, 5), p=0.7
                        ),
                        # TODO: VectorRandomElasticDeformation
                        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
                        tio.RandomNoise(mean=0.0, std=(0.0, 0.25), p=0.3),
                        tio.RandomBlur(std=(0.5, 1.5), p=0.2),
                        tio.Lambda(self._clean_tensor),
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._jitter_background),
                    ]
                )
            else:
                self.transform = tio.Compose(
                    [
                        tio.Lambda(self._tensor_to_eigvals_v1),
                        tio.Lambda(self._znorm_nonzero),
                        tio.Lambda(self._resize_to_64),
                    ]
                )

    @staticmethod
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
