# MVPI

Official code for **Multi-view patch inference with frozen DINOv3 for industrial anomaly detection**, published in *Multimedia Systems*.

- **Paper:** [Springer Nature article page](https://link.springer.com/article/10.1007/s00530-026-02685-0)
- **Method:** Multi-view Patch Inference (MVPI) with a frozen DINOv3 backbone and a Multi-scale Reconstruction Fusion Transformer (MRFT).
- **Benchmarks:** MVTec AD, BTAD, and UltraChip (referred to as CAD in the code).

## Repository contents

- `main.py`, `trainer.py`: training and evaluation entry points.
- `datasets/`, `models/`, `utils/`: dataset loaders, model components, and utilities.
- `dinov3-main/`: local DINOv3 source used by `torch.hub`.
- `distances/`: precomputed distance maps used by the model.
- `scripts/`: evaluation utilities.
- `checkpoints/`: intentionally contains no model weights. If you need the trained checkpoints, please contact the corresponding author.

The `experiments/` and `vis_results/` folders and all trained checkpoints are not included in this public repository.

## Requirements

Python 3.10 or 3.11 and a CUDA-capable GPU are recommended. Install PyTorch and torchvision matching your CUDA version, then install the remaining packages used by the code:

```bash
pip install numpy scipy scikit-learn scikit-image opencv-python Pillow tqdm thop
```

The DINOv3 source is included under `dinov3-main/`. Download the pretrained DINOv3 checkpoint required for your selected architecture from the official DINOv3 distribution and place it in the PyTorch Hub checkpoints directory, normally `~/.cache/torch/hub/checkpoints/`. For `dinov3_vitb16`, the expected filename is `dinov3_vitb16_pretrain.pth`.

Set `DINOV3_HUB_DIR` if the DINOv3 source is stored elsewhere, and `DINOV3_CHECKPOINT_DIR` if its pretrained checkpoint is stored outside the default PyTorch Hub directory.

## Dataset layout

Download the datasets from their respective official sources and update `--data_path` to the dataset root. The loaders expect the original dataset directory structures for MVTec AD, BTAD, and the CAD/UltraChip dataset. Dataset files are not included here.

## Training

Run training from the repository root. The default configuration trains all classes in the selected dataset:

```bash
python main.py --mode train --dataset mvtec --data_path /path/to/mvtec
```

Other supported datasets are `btad` and `CAD` (UltraChip). To train a selected class only:

```bash
python main.py --mode train --dataset mvtec --data_path /path/to/mvtec --classes bottle
```

The main options include `--num_epochs`, `--lr`, `--seed`, `--backbone_arch`, `--target_layers`, `--use_mrft`, `--use_mvpi`, `--overlap_patch_size`, and `--overlap_size`. Run `python main.py --help` for the full argument list. Training checkpoints are written under `checkpoints/` by default; this directory is ignored by Git apart from its placeholder file.

## Testing

To evaluate a trained checkpoint, set `--mode test` and provide the checkpoint directory with `--checkpoint`:

```bash
python main.py --mode test --dataset mvtec --data_path /path/to/mvtec --classes bottle --checkpoint checkpoints/MVTEC
```

For the MVTec multi-GPU evaluation utility:

```bash
python scripts/eval_mvtec.py --checkpoint_root checkpoints/MVTEC --data_path /path/to/mvtec
```

Use `--device cuda:0` to run that script on one selected GPU. Results are printed to the console; the evaluation script also writes a JSON summary.

## Checkpoints

No trained MVPI model weights are distributed in this repository. Please contact the corresponding author to request access to the trained weights. The DINOv3 pretrained backbone checkpoint is a separate dependency and must be obtained from the official DINOv3 distribution.

## Citation

```bibtex
@article{Wang2026MVPI,
  title   = {Multi-view patch inference with frozen DINOv3 for industrial anomaly detection},
  journal = {Multimedia Systems},
  year    = {2026},
  doi     = {10.1007/s00530-026-02685-0},
  url     = {https://link.springer.com/article/10.1007/s00530-026-02685-0}
}
```
