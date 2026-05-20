import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


def load_model(model_name: str = "dinov2_vits14_reg", device: str | None = None) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)
    return model


class DINOv2Extractor:
    """Wraps DINOv2 for batched CLS and patch token extraction."""

    def __init__(self, model_name: str = "dinov2_vits14_reg", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = load_model(model_name, device)
        self.model_name = model_name

    @torch.no_grad()
    def extract_cls(self, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_features [N, D], labels [N])."""
        all_cls, all_labels = [], []
        for images, labels in tqdm(dataloader, desc="Extracting CLS"):
            images = images.to(self.device)
            out = self.model.forward_features(images)
            all_cls.append(out["x_norm_clstoken"].cpu())
            all_labels.append(labels)
        return torch.cat(all_cls), torch.cat(all_labels)

    @torch.no_grad()
    def extract_patches(self, dataloader, dtype=torch.float16) -> torch.Tensor:
        """Returns patch features [N, 256, D] in given dtype."""
        all_patches = []
        for images, _ in tqdm(dataloader, desc="Extracting patches"):
            images = images.to(self.device)
            out = self.model.forward_features(images)
            # x_norm_patchtokens already excludes CLS and register tokens
            patches = out["x_norm_patchtokens"].to(dtype).cpu()
            all_patches.append(patches)
        return torch.cat(all_patches)

    @torch.no_grad()
    def extract_both(self, dataloader, patch_dtype=torch.float16):
        """Returns (cls [N, D], patches [N, 256, D], labels [N])."""
        all_cls, all_patches, all_labels = [], [], []
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            out = self.model.forward_features(images)
            all_cls.append(out["x_norm_clstoken"].cpu())
            all_patches.append(out["x_norm_patchtokens"].to(patch_dtype).cpu())
            all_labels.append(labels)
        return torch.cat(all_cls), torch.cat(all_patches), torch.cat(all_labels)

    def save_features(
        self,
        cls: torch.Tensor,
        labels: torch.Tensor,
        output_dir: str | Path,
        patches: torch.Tensor | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cls, output_dir / "eurosat_cls.pt")
        torch.save(labels, output_dir / "eurosat_labels.pt")
        if patches is not None:
            torch.save(patches, output_dir / "eurosat_patches.pt")
        print(f"Saved to {output_dir}")
        print(f"  CLS:    {cls.shape}  ({cls.element_size() * cls.numel() / 1e6:.1f} MB)")
        if patches is not None:
            print(f"  Patches: {patches.shape}  ({patches.element_size() * patches.numel() / 1e6:.1f} MB)")

    @staticmethod
    def load_features(output_dir: str | Path, load_patches: bool = False):
        output_dir = Path(output_dir)
        cls = torch.load(output_dir / "eurosat_cls.pt")
        labels = torch.load(output_dir / "eurosat_labels.pt")
        if load_patches:
            patches = torch.load(output_dir / "eurosat_patches.pt")
            return cls, labels, patches
        return cls, labels
