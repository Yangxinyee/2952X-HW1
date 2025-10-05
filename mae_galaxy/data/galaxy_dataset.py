from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from PIL import Image


class GalaxyDataset(Dataset):
    """Simple image-folder dataset for galaxy images.

    Expects structure:
        root/
          train|val|test/
            class_a/ *.jpg
            class_b/ *.jpg
    
    If labels are not available, classes can be a single folder.
    """

    def __init__(self, root: str, split: str, transform=None) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Collect (path, label) pairs
        class_dirs: List[Path] = [p for p in split_dir.iterdir() if p.is_dir()]
        class_dirs.sort()
        if len(class_dirs) == 0:
            # flat folder
            images = sorted([p for p in split_dir.rglob("*.jpg")])
            self.samples = [(p, 0) for p in images]
            self.classes = ["unknown"]
        else:
            self.classes = [p.name for p in class_dirs]
            self.samples = []
            for class_index, class_dir in enumerate(class_dirs):
                for img_path in sorted(class_dir.rglob("*.jpg")):
                    self.samples.append((img_path, class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label




