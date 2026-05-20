import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_eurosat(root: str = "./data/eurosat", batch_size: int = 256, split: str = "train") -> tuple[DataLoader, list[str]]:
    """Load EuroSAT-RGB. Falls back to HuggingFace datasets if torchvision fails."""
    transform = get_transform(224)
    try:
        from torchvision.datasets import EuroSAT
        dataset = EuroSAT(root=root, transform=transform, download=True)
        classes = dataset.classes
    except Exception as e:
        print(f"torchvision EuroSAT failed ({e}), trying HuggingFace...")
        dataset = _EuroSATHF(transform=transform)
        classes = dataset.classes

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader, classes


class _EuroSATHF(Dataset):
    """HuggingFace fallback for EuroSAT-RGB."""

    def __init__(self, transform=None):
        from datasets import load_dataset
        raw = load_dataset("blanchon/EuroSAT_RGB", split="train", trust_remote_code=True)
        self._data = raw
        self.transform = transform
        self.classes = raw.features["label"].names

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data[idx]
        img = sample["image"].convert("RGB")
        label = sample["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


class LoveDADataset(Dataset):
    """
    LoveDA dataset loader (local files from Zenodo).

    Expected structure:
        root/
          Train/Urban/images_png/*.png
          Train/Urban/masks_png/*.png
          Train/Rural/images_png/*.png
          ...
          Val/Urban/...

    Classes (0-indexed): background, building, road, water, barren, forest, agriculture
    """

    CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]

    def __init__(self, root: str, split: str = "Train", regions: list[str] | None = None, transform=None):
        from PIL import Image

        root = Path(root)
        if regions is None:
            regions = ["Urban", "Rural"]

        self.transform = transform
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []

        for region in regions:
            img_dir = root / split / region / "images_png"
            mask_dir = root / split / region / "masks_png"
            if not img_dir.exists():
                raise FileNotFoundError(
                    f"LoveDA not found at {img_dir}.\n"
                    "Download from https://doi.org/10.5281/zenodo.5706578 and extract to the data/loveda directory."
                )
            imgs = sorted(img_dir.glob("*.png"))
            masks = sorted(mask_dir.glob("*.png"))
            assert len(imgs) == len(masks), f"Mismatch: {len(imgs)} images vs {len(masks)} masks"
            self.image_paths.extend(imgs)
            self.mask_paths.extend(masks)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        mask = torch.from_numpy(np.array(mask)).long() - 1  # 1-indexed → 0-indexed, background=0

        if self.transform:
            img = self.transform(img)

        return img, mask


def load_loveda(
    root: str = "./data/loveda",
    split: str = "Train",
    batch_size: int = 16,
    image_size: int = 224,
) -> DataLoader:
    transform = get_transform(image_size)
    dataset = LoveDADataset(root=root, split=split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def load_ucmerced(root: str = "./data/ucmerced", batch_size: int = 64) -> DataLoader:
    """UCMerced land-use dataset via torchgeo."""
    try:
        from torchgeo.datasets import UCMerced
        transform = get_transform(224)
        dataset = UCMerced(root=root, download=True, transforms=None)
        # torchgeo returns dicts; wrap to standard (image, label)
        return DataLoader(
            _TorchGeoWrapper(dataset, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
    except ImportError:
        raise ImportError("torchgeo required for UCMerced: pip install torchgeo")


class _TorchGeoWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        # torchgeo returns CHW uint8 tensor
        from torchvision.transforms.functional import to_pil_image
        img = to_pil_image(img)
        if self.transform:
            img = self.transform(img)
        return img, sample["label"]
