"""
Compute mean and std statistics for galaxy dataset.
Usage:
    python compute_dataset_stats.py --use_hf  # for HuggingFace dataset
    python compute_dataset_stats.py --data_root ./data  # for local dataset
"""
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mae_galaxy.data.galaxy_dataset import GalaxyDataset
from mae_galaxy.data.hf_galaxy_dataset import HFGalaxyDataset, create_hf_splits


def compute_mean_std(dataloader):
    """
    Compute channel-wise mean and std across entire dataset.
    Uses Welford's online algorithm for numerical stability.
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    print("Computing mean...")
    # First pass: compute mean
    for imgs, _ in tqdm(dataloader, desc="Pass 1/2 - Mean"):
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)  # (B, C, H*W)
        mean += imgs.mean(dim=[0, 2]) * batch_size
        total_samples += batch_size

    mean = mean / total_samples

    print("Computing std...")
    # Second pass: compute std
    for imgs, _ in tqdm(dataloader, desc="Pass 2/2 - Std"):
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)  # (B, C, H*W)
        std += ((imgs - mean.view(1, 3, 1)) ** 2).sum(dim=[0, 2])

    std = torch.sqrt(std / (total_samples * 224 * 224))

    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--use_hf", action="store_true", help="Use HuggingFace dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Minimal transform: only resize and convert to tensor
    # Do NOT normalize yet, we need raw pixel values
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])

    print("Loading dataset...")
    if args.use_hf:
        train_split, _, _ = create_hf_splits()
        dataset = HFGalaxyDataset(train_split, transform=transform)
    else:
        dataset = GalaxyDataset(args.data_root, split="train", transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)} images")

    mean, std = compute_mean_std(dataloader)

    print("\n" + "="*60)
    print("Galaxy Dataset Statistics (after ToTensor, range [0,1]):")
    print("="*60)
    print(f"Mean (RGB): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"Std  (RGB): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("="*60)
    print("\nUsage in transforms:")
    print("transforms.Normalize(")
    print(f"    mean=[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}],")
    print(f"    std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print(")")
    print("\nFor comparison, ImageNet stats:")
    print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")


if __name__ == "__main__":
    main()