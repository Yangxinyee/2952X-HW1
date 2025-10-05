import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import time
import json

from mae_galaxy.data.galaxy_dataset import GalaxyDataset
from mae_galaxy.data.hf_galaxy_dataset import HFGalaxyDataset, create_hf_splits
from mae_galaxy.models.vit_baseline import get_vit_encoder
from mae_galaxy.models.mae_model import MAEModel
from mae_galaxy.utils.logger import CSVLogger
from mae_galaxy.utils.visualization import plot_curve_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear Probe")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--encoder", type=str, choices=["random", "imagenet", "mae"], default="random")
    parser.add_argument("--mae_ckpt", type=str, default="", help="Path to MAE checkpoint when --encoder mae")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_hf", action="store_true", help="Use Hugging Face dataset loader")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./outputs/probe", help="Where to save probe checkpoints")
    parser.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N epochs")
    parser.add_argument("--keep_best", action="store_true", help="Save best validation accuracy to best.pt")
    parser.add_argument("--with_timestamp", action="store_true", help="Append timestamp to output_dir name")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder during training (default: True for linear probing)")
    parser.add_argument("--finetune", action="store_true", help="Finetune encoder (unfrozen) instead of linear probing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training transforms with data augmentation
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])
    
    # Validation transforms without augmentation
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    if args.use_hf:
        train_split, val_split, test_split = create_hf_splits()
        train_ds = HFGalaxyDataset(train_split, transform=train_tf)
        val_ds = HFGalaxyDataset(val_split, transform=val_tf)
        test_ds = HFGalaxyDataset(test_split, transform=val_tf)  # Use val_tf (no augmentation)
        num_classes = len(train_ds.classes) if train_ds.classes is not None else 10
    else:
        train_ds = GalaxyDataset(args.data_root, split="train", transform=train_tf)
        val_ds = GalaxyDataset(args.data_root, split="val", transform=val_tf)
        test_ds = GalaxyDataset(args.data_root, split="test", transform=val_tf)
        num_classes = len(train_ds.classes)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Determine if encoder should be frozen (default True, unless --finetune is specified)
    freeze_encoder = not args.finetune if not args.freeze_encoder else args.freeze_encoder

    if args.encoder in ("random", "imagenet"):
        encoder, feat_dim = get_vit_encoder(args.encoder)
        encoder.to(device)
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

        def extract_feats(x: torch.Tensor) -> torch.Tensor:
            if freeze_encoder:
                with torch.no_grad():
                    feats = encoder(x)
            else:
                feats = encoder(x)
            if feats.dim() == 3:
                feats = feats[:, 0] if feats.size(1) > 1 else feats.mean(dim=1)
            return feats
    else:
        if not args.mae_ckpt:
            raise ValueError("--mae_ckpt is required when --encoder mae")
        mae = MAEModel()
        ckpt = torch.load(args.mae_ckpt, map_location="cpu")
        mae.load_state_dict(ckpt["model"], strict=False)
        mae.to(device)
        if freeze_encoder:
            for p in mae.parameters():
                p.requires_grad = False
            mae.eval()
        feat_dim = 768

        def extract_feats(x: torch.Tensor) -> torch.Tensor:
            if freeze_encoder:
                with torch.no_grad():
                    _, _, enc_tokens = mae(x)
            else:
                _, _, enc_tokens = mae(x)
            return enc_tokens.mean(dim=1)

    classifier = nn.Linear(feat_dim, num_classes).to(device)

    # If finetuning, optimize both encoder and classifier
    if args.finetune:
        if args.encoder in ("random", "imagenet"):
            trainable_params = list(encoder.parameters()) + list(classifier.parameters())
        else:
            trainable_params = list(mae.parameters()) + list(classifier.parameters())
        optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        optim = torch.optim.AdamW(classifier.parameters(), lr=args.lr)

    loss_fn = nn.CrossEntropyLoss()

    # Prepare output directory
    if args.with_timestamp:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"{args.output_dir}_{stamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save flags/hparams
    try:
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception:
        pass

    def run_epoch(dataloader, train: bool, epoch_idx: int):
        if train:
            classifier.train()
            if args.finetune:
                if args.encoder in ("random", "imagenet"):
                    encoder.train()
                else:
                    mae.train()
        else:
            classifier.eval()
            if args.encoder in ("random", "imagenet"):
                encoder.eval()
            else:
                mae.eval()
        total_correct = 0
        total = 0
        total_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            feats = extract_feats(imgs)
            logits = classifier(feats)
            loss = loss_fn(logits, labels)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / max(1, total)
                avg_acc = total_correct / max(1, total)
                msg = f"Epoch {epoch_idx+1} Step {batch_idx+1} | Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}"
                print(msg, end='\r', flush=True)
        return total_loss / max(1, total), total_correct / max(1, total)

    best_acc = -1.0
    log_path = str(out_dir / "probe_log.csv")
    logger = CSVLogger(log_path, ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"])
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(train_dl, train=True, epoch_idx=epoch)
        val_loss, val_acc = run_epoch(val_dl, train=False, epoch_idx=epoch)
        test_loss, test_acc = run_epoch(test_dl, train=False, epoch_idx=epoch)
        print()
        print(f"Epoch {epoch+1} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | test loss {test_loss:.4f} acc {test_acc:.4f}")
        logger.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc})

        # Save periodic checkpoint
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            ckpt = {
                "classifier": classifier.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "encoder_type": args.encoder,
                "feat_dim": feat_dim,
                "num_classes": num_classes,
                "mae_ckpt": args.mae_ckpt if args.encoder == "mae" else "",
            }
            torch.save(ckpt, out_dir / f"epoch_{epoch+1}.pt")

        # Save best by validation accuracy
        if args.keep_best and val_acc > best_acc:
            best_acc = val_acc
            best = {
                "classifier": classifier.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "encoder_type": args.encoder,
                "feat_dim": feat_dim,
                "num_classes": num_classes,
                "mae_ckpt": args.mae_ckpt if args.encoder == "mae" else "",
            }
            torch.save(best, out_dir / "best.pt")
            print(f"[best] Updated best probe (acc {val_acc:.4f}) at epoch {epoch+1}")
    print("Finished linear probe.")
    # Plot curves
    try:
        plot_curve_from_csv(log_path, x="epoch", y="val_acc", out_path=str(out_dir / "val_acc_curve.png"), title="Linear Probe Val Accuracy")
        plot_curve_from_csv(log_path, x="epoch", y="test_acc", out_path=str(out_dir / "test_acc_curve.png"), title="Linear Probe Test Accuracy")
        plot_curve_from_csv(log_path, x="epoch", y="train_loss", out_path=str(out_dir / "train_loss_curve.png"), title="Linear Probe Train Loss")
    except Exception:
        pass


if __name__ == "__main__":
    main()


