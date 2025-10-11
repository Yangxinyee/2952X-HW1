## MAE Galaxy Dataset Project

### Environment Requirements
- **GPU**: NVIDIA RTX A6000 (or similar)
- **CUDA**: 12.1+ (tested with 12.2)
- **Python**: 3.10
- **Key Dependencies**:
  - PyTorch: 2.8.0 (with CUDA 12.8)
  - torchvision: 0.23.0
  - timm: 1.0.19
  - datasets: 4.1.0
  - numpy: 2.2.6
  - pandas: 2.3.2
  - matplotlib: 3.10.6

### Setup
1. Clone repository: `git clone https://github.com/Yangxinyee/2952X-HW1.git`
2. Navigate to project: `cd 2952X-HW1`
3. Create env: `conda create -n mae-hw1 python=3.10`
4. Activate: `conda activate mae-hw1`
5. Install: `pip install -r requirements.txt`

### Project structure
```
mae_galaxy/
├── data/
│   └── galaxy_dataset.py      # Dataset loading and preprocessing
├── models/
│   ├── mae_model.py          # MAE implementation
│   └── vit_baseline.py       # ViT baselines
├── training/
│   ├── train_mae.py          # MAE pretraining
│   └── linear_probe.py       # Linear probing evaluation
├── utils/
│   ├── visualization.py      # Plotting and visualization
│   └── metrics.py            # Evaluation metrics
├── experiments/
│   └── run_experiments.py    # Main experiment runner
├── requirements.txt
├── README.md
└── main.py                   # Entry point
```

### Entry points
- Pretrain MAE: `python -m mae_galaxy.training.train_mae --help`
- Linear probe: `python -m mae_galaxy.training.linear_probe --help`
- Experiments: `python -m mae_galaxy.experiments.run_experiments --help`

### Example
- method_comparison
```bash
python -m mae_galaxy.experiments.run_experiments --experiment method_comparison --epochs_mae 200 --masked_only_loss --init_encoder imagenet_mae --epochs_probe 100 --mask_ratio 0.75 --decoder_depth 8 --use_hf --keep_best
```

- masking_ablation
```bash
python -m mae_galaxy.experiments.run_experiments --experiment masking_ablation --masked_only_loss --use_hf --init_encoder imagenet_mae --epochs_mae 200 --epochs_probe 150 --output_dir ./outputs --save_every 100 --keep_best
```

- decoder_depth_ablation
```bash
python -m mae_galaxy.experiments.run_experiments --experiment decoder_depth_ablation --masked_only_loss --use_hf --init_encoder imagenet_mae --epochs_mae 200 --epochs_probe 150 --output_dir ./outputs --save_every 100 --keep_best
```

- encoder_freezing_comparison
```bash
python -m mae_galaxy.experiments.run_experiments --experiment encoder_freezing_comparison --output_dir ./outputs --epochs_mae 200 --epochs_probe 150 --use_hf --keep_best --masked_only_loss --save_every 200
```











