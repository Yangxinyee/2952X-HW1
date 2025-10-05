from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import pandas as pd


def show_reconstructions(images: torch.Tensor, recons: torch.Tensor, n: int = 4) -> None:
    n = min(n, images.size(0))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i].permute(1, 2, 0).detach().cpu().numpy())
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_curve_from_csv(csv_path: str, x: str, y: str, out_path: Optional[str] = None, title: str = "") -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x], df[y], linewidth=2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()



