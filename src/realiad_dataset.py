"""Single-view PyTorch dataset for Real-IAD classification and segmentation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFECT_TO_ID = {
    "AK": 0,
    "BX": 1,
    "CH": 2,
    "HS": 3,
    "PS": 4,
    "QS": 5,
    "YW": 6,
    "ZW": 7,
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RealIADSingleViewDataset(Dataset):
    """Load one Real-IAD camera image and its ground-truth mask per item."""

    def __init__(self, csv_path: str | Path, train: bool = False) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_absolute():
            self.csv_path = self.repo_root / self.csv_path

        with self.csv_path.open(newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.repo_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        if row["mask_path"]:
            mask_path = self.repo_root / row["mask_path"]
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

        if image.size != (256, 256):
            image = image.resize((256, 256), Image.Resampling.BILINEAR)
            mask = mask.resize((256, 256), Image.Resampling.NEAREST)

        if self.train and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.train and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)[None]
        image_tensor = torch.from_numpy(image_array)
        mask_tensor = torch.from_numpy(mask_array)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        is_defective = row["label"] == "NG"
        defect_type = row["defect_type"]
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(float(is_defective), dtype=torch.float32),
            "defect_type_id": torch.tensor(
                DEFECT_TO_ID.get(defect_type, -1), dtype=torch.long
            ),
            "defect_type": defect_type or "OK",
            "category": row["category"],
            "sample_id": row["sample_id"],
            "camera": row["camera"],
            "image_path": row["image_path"],
        }


def make_dataloader(
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for train, val, or test."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")

    dataset = RealIADSingleViewDataset(
        f"data/splits/{split}.csv",
        train=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    loader = make_dataloader("train", batch_size=4)
    batch = next(iter(loader))
    print("images:", tuple(batch["image"].shape))
    print("masks: ", tuple(batch["mask"].shape))
    print("labels:", tuple(batch["label"].shape))
