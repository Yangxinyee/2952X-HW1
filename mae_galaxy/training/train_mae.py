import argparse
import time
from pathlib import Path
import json
import math

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from mae_galaxy.data.galaxy_dataset import GalaxyDataset
from mae_galaxy.data.hf_galaxy_dataset import HFGalaxyDataset, create_hf_splits
from mae_galaxy.models.mae_model import MAEModel
from mae_galaxy.utils.logger import CSVLogger
from mae_galaxy.utils.visualization import plot_curve_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="MAE Pretraining")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/mae")
    parser.add_argument("--use_hf", action="store_true", help="Use Hugging Face dataset loader")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Base learning rate (MAE paper: 1.5e-4 * batch_size / 256)")
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=200, help="Save a checkpoint every N epochs")
    parser.add_argument("--keep_best", action="store_true", help="Track and save best epoch loss to best.pt")
    parser.add_argument("--with_timestamp", action="store_true", help="Append timestamp to output_dir name")
    parser.add_argument("--masked_only_loss", action="store_true", help="Compute loss only on masked patches (MAE convention)")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs for cosine LR")
    parser.add_argument("--init_encoder", type=str, choices=["scratch", "imagenet_mae"], default="scratch", help="Initialize encoder weights")
    parser.add_argument("--init_ckpt", type=str, default="", help="Path to MAE ImageNet checkpoint (if --init_encoder imagenet_mae)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.with_timestamp:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"{args.output_dir}_{stamp}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Save training flags/hparams
    try:
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception:
        pass

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])

    if args.use_hf:
        train_split, _, _ = create_hf_splits()
        ds = HFGalaxyDataset(train_split, transform=train_tf)
    else:
        ds = GalaxyDataset(args.data_root, split="train", transform=train_tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = MAEModel(mask_ratio=args.mask_ratio, decoder_depth=args.decoder_depth).to(device)

    # Optionally initialize encoder from provided MAE (ImageNet) checkpoint
    if args.init_encoder == "imagenet_mae":
        state = None
        if args.init_ckpt:
            ckpt = torch.load(args.init_ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt)
        else:
            # Try loading MAE-pretrained ViT from timm (pulls from HF under the hood)
            try:
                import timm
                mae_vit = timm.create_model("vit_base_patch16_224.mae", pretrained=True)
                state = mae_vit.state_dict()
                # Prefix encoder. to match our model module names
                prefixed = {}
                for k, v in state.items():
                    if k.startswith(("patch_embed", "pos_embed", "cls_token", "blocks", "norm")):
                        prefixed[f"encoder.{k}"] = v
                state = prefixed
            except Exception as e:
                raise ValueError("Failed to load MAE init from timm; please provide --init_ckpt path") from e
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded MAE init with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # Cosine LR with warmup (per-epoch) - modified to maintain minimum LR
    def lr_lambda(current_epoch: int) -> float:
        if args.epochs <= 0:
            return 1.0
        if current_epoch < args.warmup_epochs:
            return float(current_epoch + 1) / max(1, args.warmup_epochs)
        progress = (current_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        # Maintain minimum learning rate of 1e-6 instead of going to 0
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        min_lr_ratio = 1e-6 / args.lr  # Minimum LR as ratio of initial LR
        return max(min_lr_ratio, cosine_factor)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    loss_fn = torch.nn.MSELoss(reduction='none')

    model.train()
    global_step = 0
    best_loss = float("inf")
    log_path = str(Path(args.output_dir) / "train_log.csv")
    logger = CSVLogger(log_path, ["epoch", "loss", "lr"]) 

    # Loss-space normalization (Galaxy dataset stats) applied ONLY for loss computation
    # Computed from training set: mean=[0.168, 0.163, 0.159], std=[0.123, 0.113, 0.106]
    galaxy_mean = torch.tensor([0.167775, 0.162880, 0.159137], device=device).view(1, 3, 1, 1)
    galaxy_std = torch.tensor([0.122556, 0.113329, 0.106091], device=device).view(1, 3, 1, 1)

    def normalize_for_loss(x: torch.Tensor) -> torch.Tensor:
        return (x - galaxy_mean) / galaxy_std
    for epoch in range(args.epochs):
        running_loss = 0.0
        num_samples = 0
        for batch_idx, (imgs, _) in enumerate(dl):
            imgs = imgs.to(device)
            recons, mask, _ = model(imgs)

            if args.masked_only_loss:
                # Compute MSE on masked patches only
                with torch.no_grad():
                    target_patches = model.patchify(normalize_for_loss(imgs))  # (B, N, P)
                pred_patches = model.patchify(normalize_for_loss(recons))
                per_patch_mse = (pred_patches - target_patches) ** 2  # (B, N, P)
                per_patch_mse = per_patch_mse.mean(dim=-1)  # (B, N)
                masked_mse = (per_patch_mse * mask).sum() / mask.sum().clamp_min(1.0)
                loss = masked_mse
            else:
                # Full-image MSE
                loss_map = loss_fn(normalize_for_loss(recons), normalize_for_loss(imgs))  # (B, C, H, W)
                loss = loss_map.mean()
            optim.zero_grad()
            loss.backward()
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            global_step += 1

            running_loss += loss.item() * imgs.size(0)
            num_samples += imgs.size(0)

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / max(1, num_samples)
                msg = f"Epoch {epoch+1} Step {batch_idx+1} | Avg Loss: {avg_loss:.4f}"
                print(msg, end='\r', flush=True)

        epoch_loss = running_loss / max(1, num_samples)
        print()  # ensure newline after single-line progress
        current_lr = optim.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} completed | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")
        logger.log({"epoch": epoch + 1, "loss": epoch_loss, "lr": current_lr})
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            ckpt_path = Path(args.output_dir) / f"epoch_{epoch+1}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)
        if args.keep_best and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = Path(args.output_dir) / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "loss": epoch_loss}, best_path)
            print(f"[best] Updated best checkpoint at epoch {epoch+1} (loss {epoch_loss:.4f})")

        # Step scheduler after each epoch
        scheduler.step()

    # Plot loss curve
    try:
        plot_curve_from_csv(log_path, x="epoch", y="loss", out_path=str(Path(args.output_dir) / "loss_curve.png"), title="MAE Pretrain Loss")
    except Exception:
        pass
    print("Finished MAE pretraining.")


if __name__ == "__main__":
    main()


