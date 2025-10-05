from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset, Dataset as HFDataset, ClassLabel


class HFGalaxyDataset(Dataset):
    """Wrap a Hugging Face dataset split for Galaxy10 DECals.

    Expects columns: 'image' (PIL or array) and 'label' (int).
    """

    def __init__(self, split_ds: HFDataset, transform=None) -> None:
        super().__init__()
        self.ds = split_ds
        self.transform = transform

        # Infer number of classes if available
        self.classes = None
        if "label" in self.ds.features and isinstance(self.ds.features["label"], ClassLabel):
            self.classes = list(self.ds.features["label"].names)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        item = self.ds[index]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        if "label" in item:
            label = int(item["label"])
        elif "labels" in item:
            label = int(item["labels"])
        elif "target" in item:
            label = int(item["target"])
        return img, label


def create_hf_splits(dataset_name: str = "matthieulel/galaxy10_decals", seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15):
    full = load_dataset(dataset_name, split="train")
    # If dataset provides train/test already, concatenate for custom split
    try:
        test = load_dataset(dataset_name, split="test")
        full = full.concatenate(test)
    except Exception:
        pass

    # Determine if we can stratify (only when ClassLabel)
    stratify_col = None
    if "label" in full.features and isinstance(full.features["label"], ClassLabel):
        stratify_col = "label"

    # First split train vs temp
    if stratify_col is not None:
        train_testval = full.train_test_split(test_size=1 - train_ratio, seed=seed, stratify_by_column=stratify_col)
    else:
        train_testval = full.train_test_split(test_size=1 - train_ratio, seed=seed)
    train_split = train_testval["train"]
    temp_split = train_testval["test"]
    # Split temp into val and test
    val_size = val_ratio / (1 - train_ratio)
    if stratify_col is not None:
        val_test = temp_split.train_test_split(test_size=1 - val_size, seed=seed, stratify_by_column=stratify_col)
    else:
        val_test = temp_split.train_test_split(test_size=1 - val_size, seed=seed)
    val_split = val_test["train"]
    test_split = val_test["test"]
    return train_split, val_split, test_split


