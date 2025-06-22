import os
import torch
import torchio as tio
from torch.utils.data import Dataset


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
                        tio.Lambda(self._clean),
                        tio.ZNormalization(),
                        tio.RandomGamma(log_gamma=(-0.3, 0.3)),  # contrast/brightness
                        tio.RandomNoise(mean=0.0, std=(0, 0.25)),
                        tio.RandomBlur(std=(0.5, 1.5)),
                        tio.CropOrPad((64, 64, 64)),
                    ]
                )
            else:
                self.transform = tio.Compose(
                    [
                        tio.Lambda(self._clean),
                        tio.ZNormalization(),
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
                transformed1 = self.transform(subject)
                transformed2 = self.transform(subject)

                data1 = transformed1["dti"].data.float()
                data2 = transformed2["dti"].data.float()

                return data1, data2

            except Exception as e:
                print(
                    f"[Warning] Failed to process {nii_path}: {e}, trying next index..."
                )
                idx = (idx + 1) % len(self.sample_paths)

        raise RuntimeError(
            f"[Error] Failed to load a valid sample after {max_attempts} attempts."
        )
