import argparse
import subprocess
from pathlib import Path
import time


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiments")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment", type=str, choices=[
        "method_comparison",
        "masking_ablation",
        "decoder_depth_ablation",
        "encoder_freezing_comparison",
    ], default="method_comparison")
    parser.add_argument("--use_hf", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--epochs_mae", type=int, default=200)
    parser.add_argument("--epochs_probe", type=int, default=100)
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--keep_best", action="store_true")
    parser.add_argument("--masked_only_loss", action="store_true", help="Use masked-only loss for MAE training")
    parser.add_argument("--init_encoder", type=str, choices=["scratch", "imagenet_mae"], default="scratch")
    parser.add_argument("--init_ckpt", type=str, default="")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    hf_flag = " --use_hf" if args.use_hf else ""
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.experiment == "method_comparison":
        mae_dir = f"{args.output_dir}/mae_{stamp}"
        run(
            f"python -m mae_galaxy.training.train_mae"
            f" --data_root {args.data_root}"
            f" --output_dir {mae_dir}"
            f" --epochs {args.epochs_mae}"
            f" --mask_ratio {args.mask_ratio}"
            f" --decoder_depth {args.decoder_depth}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" {'--masked_only_loss' if args.masked_only_loss else ''}"
            f" --init_encoder {args.init_encoder}"
            f" {'--init_ckpt ' + args.init_ckpt if (args.init_encoder=='imagenet_mae' and args.init_ckpt) else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        rand_dir = f"{args.output_dir}/linear_random_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder random"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {rand_dir}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        img_dir = f"{args.output_dir}/linear_imgnet_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder imagenet"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {img_dir}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        # probe MAE encoder if best checkpoint exists
        mae_best = Path(mae_dir) / "best.pt"
        mae_ckpt = mae_best if mae_best.exists() else (Path(mae_dir) / "epoch_1.pt")
        mae_probe_dir = f"{args.output_dir}/linear_mae_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder mae"
            f" --mae_ckpt {mae_ckpt}"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {mae_probe_dir}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )
    elif args.experiment == "masking_ablation":
        for r in [0.5, 0.75, 0.9]:
            mae_dir = f"{args.output_dir}/mae_r{r}_{stamp}"
            run(
                f"python -m mae_galaxy.training.train_mae"
                f" --data_root {args.data_root}"
                f" --output_dir {mae_dir}"
                f" --mask_ratio {r}"
                f" --epochs {args.epochs_mae}"
                f" --decoder_depth {args.decoder_depth}"
                f" --save_every {args.save_every}"
                f" {'--keep_best' if args.keep_best else ''}"
                f" {'--masked_only_loss' if args.masked_only_loss else ''}"
                f" --init_encoder {args.init_encoder}"
                f" {'--init_ckpt ' + args.init_ckpt if (args.init_encoder=='imagenet_mae' and args.init_ckpt) else ''}"
                f" --log_interval {args.log_interval}{hf_flag}"
            )

            # Linear probe the trained MAE encoder
            mae_best = Path(mae_dir) / "best.pt"
            mae_ckpt = mae_best if mae_best.exists() else (Path(mae_dir) / f"epoch_{args.epochs_mae}.pt")
            probe_dir = f"{args.output_dir}/linear_mae_r{r}_{stamp}"
            run(
                f"python -m mae_galaxy.training.linear_probe"
                f" --data_root {args.data_root}"
                f" --encoder mae"
                f" --mae_ckpt {mae_ckpt}"
                f" --epochs {args.epochs_probe}"
                f" --output_dir {probe_dir}"
                f" --save_every {args.save_every}"
                f" {'--keep_best' if args.keep_best else ''}"
                f" --log_interval {args.log_interval}{hf_flag}"
            )
    elif args.experiment == "decoder_depth_ablation":
        for d in [2, 4, 8]:
            mae_dir = f"{args.output_dir}/mae_d{d}_{stamp}"
            run(
                f"python -m mae_galaxy.training.train_mae"
                f" --data_root {args.data_root}"
                f" --output_dir {mae_dir}"
                f" --decoder_depth {d}"
                f" --epochs {args.epochs_mae}"
                f" --mask_ratio {args.mask_ratio}"
                f" --save_every {args.save_every}"
                f" {'--keep_best' if args.keep_best else ''}"
                f" {'--masked_only_loss' if args.masked_only_loss else ''}"
                f" --init_encoder {args.init_encoder}"
                f" {'--init_ckpt ' + args.init_ckpt if (args.init_encoder=='imagenet_mae' and args.init_ckpt) else ''}"
                f" --log_interval {args.log_interval}{hf_flag}"
            )

            # Linear probe the trained MAE encoder
            mae_best = Path(mae_dir) / "best.pt"
            mae_ckpt = mae_best if mae_best.exists() else (Path(mae_dir) / f"epoch_{args.epochs_mae}.pt")
            probe_dir = f"{args.output_dir}/linear_mae_d{d}_{stamp}"
            run(
                f"python -m mae_galaxy.training.linear_probe"
                f" --data_root {args.data_root}"
                f" --encoder mae"
                f" --mae_ckpt {mae_ckpt}"
                f" --epochs {args.epochs_probe}"
                f" --output_dir {probe_dir}"
                f" --save_every {args.save_every}"
                f" {'--keep_best' if args.keep_best else ''}"
                f" --log_interval {args.log_interval}{hf_flag}"
            )
    elif args.experiment == "encoder_freezing_comparison":
        # First train MAE model
        mae_dir = f"{args.output_dir}/mae_{stamp}"
        run(
            f"python -m mae_galaxy.training.train_mae"
            f" --data_root {args.data_root}"
            f" --output_dir {mae_dir}"
            f" --epochs {args.epochs_mae}"
            f" --mask_ratio {args.mask_ratio}"
            f" --decoder_depth {args.decoder_depth}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" {'--masked_only_loss' if args.masked_only_loss else ''}"
            f" --init_encoder {args.init_encoder}"
            f" {'--init_ckpt ' + args.init_ckpt if (args.init_encoder=='imagenet_mae' and args.init_ckpt) else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        # Get MAE checkpoint
        mae_best = Path(mae_dir) / "best.pt"
        mae_ckpt = mae_best if mae_best.exists() else (Path(mae_dir) / f"epoch_{args.epochs_mae}.pt")

        # Test 1: Frozen encoder (standard linear probing)
        frozen_dir = f"{args.output_dir}/mae_frozen_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder mae"
            f" --mae_ckpt {mae_ckpt}"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {frozen_dir}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        # Test 2: Unfrozen encoder (fine-tuning)
        finetune_dir = f"{args.output_dir}/mae_finetune_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder mae"
            f" --mae_ckpt {mae_ckpt}"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {finetune_dir}"
            f" --finetune"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        # Also compare with ImageNet encoder
        imgnet_frozen_dir = f"{args.output_dir}/imagenet_frozen_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder imagenet"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {imgnet_frozen_dir}"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )

        imgnet_finetune_dir = f"{args.output_dir}/imagenet_finetune_{stamp}"
        run(
            f"python -m mae_galaxy.training.linear_probe"
            f" --data_root {args.data_root}"
            f" --encoder imagenet"
            f" --epochs {args.epochs_probe}"
            f" --output_dir {imgnet_finetune_dir}"
            f" --finetune"
            f" --save_every {args.save_every}"
            f" {'--keep_best' if args.keep_best else ''}"
            f" --log_interval {args.log_interval}{hf_flag}"
        )
    print("Experiments finished.")


if __name__ == "__main__":
    main()


