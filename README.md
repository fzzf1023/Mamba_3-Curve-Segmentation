<div align="center">
  <h1>Mamba-3 Chart Curve Instance Segmentation</h1>
  <p>Pure PyTorch chart curve instance segmentation built on a Mamba-3 spatial encoder.</p>
  <p>
    <a href="#installation"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="#model-variants"><img src="https://img.shields.io/badge/default-CurveSOTAQueryNet-0969da?style=flat-square" alt="Default Model"></a>
    <a href="index.html"><img src="https://img.shields.io/badge/docs-index.html-1f883d?style=flat-square" alt="Docs"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-8250df?style=flat-square" alt="License"></a>
  </p>
</div>

> This README is aligned with the current local implementation. The default training entry point is `CurveSOTAQueryNet` via `train.py --model sota`.

## Overview

This repository combines two related parts:

- A chart curve instance segmentation system with a Mamba-3 spatial encoder
- A minimal `mamba3.py` sequence-model implementation used by the spatial backbone and kept as a standalone reference

The segmentation side is the primary project in the current codebase. It includes:

- `CurveInstanceMamba3Net`: a base model with composed-mask and embedding outputs
- `CurveSOTAQueryNet`: the default query-based model for instance segmentation
- `index.html`: an English method page that can be published directly as a static site

## Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Model Variants](#model-variants)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Default Configuration](#default-configuration)
- [Evaluation](#evaluation)
- [Loss System](#loss-system)
- [Optional Modules and Caveats](#optional-modules-and-caveats)
- [Tests](#tests)
- [Repository Layout](#repository-layout)
- [Documentation Page](#documentation-page)
- [Credits](#credits)
- [License](#license)

## Highlights

- RGB + HSV + Sobel input enhancement is enabled by default in the SOTA model.
- The spatial encoder uses progressive branch schedules of `3 / 5 / 9 / 9` from `H/4` to `H/32`.
- Active branch types include local depthwise convolution, row/column Mamba scans, reverse row/column scans, and snake scans in `H / V / D45 / D135`.
- The decoder uses FPN fusion, an `H/2` stem skip, and an additive grid-suppression bias.
- The SOTA model combines a query decoder with pixel topology heads for `centerline`, `crossing`, `boundary`, `direction`, and `grid`.
- Training uses Hungarian matching, one-to-many matching, denoising queries, EMA, and loss-ramp scheduling.
- The current loss system has 16 primary terms across 5 groups, plus `topograph` and `efd` as default-disabled optional terms.
- Inference uses quality-aware scoring, crossing-aware NMS, and fragment filtering.
- The legend-guided path is implemented, but the default dataset pipeline does not emit `legend_patches`, so it is inactive by default.

## Architecture

```text
Input image
  -> RGB / HSV / Sobel enhancement
  -> Progressive Mamba encoder
     - Stage 0: H/4,  3 branches
     - Stage 1: H/8,  5 branches
     - Stage 2: H/16, 9 branches
     - Stage 3: H/32, 9 branches
  -> FPN + H/2 stem skip + additive grid bias
  -> [ Query decoder || Pixel topology heads ]

Train branch:
  Hungarian + O2M + DN + 16 primary losses + EMA + LR schedule

Infer branch:
  query logits + quality + masks + crossing-aware post-processing
```

## Model Variants

| Model | Entry point | Default status | Main outputs |
| --- | --- | --- | --- |
| Base | `CurveInstanceMamba3Net` | optional | `composed_mask`, embeddings, topology heads |
| SOTA | `CurveSOTAQueryNet` | default | query logits, query masks, quality, topology heads |

The training script defaults to:

```bash
python train.py --model sota --train_dir data/train --val_dir data/val
```

## Installation

```bash
pip install -r requirements.txt
```

Optional extras:

- `pycocotools` for official COCO AP
- `pytest` for the test suite

## Quick Start

### Training

```bash
# Base model
python train.py --model base --train_dir data/train --val_dir data/val

# Default SOTA model
python train.py --model sota --train_dir data/train --val_dir data/val \
    --batch_size 4 --img_size 512 --epochs 100 --lr 2e-4

# Resume from checkpoint
python train.py --model sota --train_dir data/train --resume checkpoints/last.pth

# Ablation: disable legend-guided queries
python train.py --model sota --train_dir data/train --val_dir data/val \
    --no_legend_queries
```

### Analysis and Utility Scripts

```bash
# Profile model size / FLOPs / throughput
python profile_model.py --model sota --img_size 512
python profile_model.py --model base --img_size 512 --device cpu

# Visualize snake offsets from a checkpoint
python visualize_offsets.py --checkpoint checkpoints/best.pth --image test.png
python visualize_offsets.py --checkpoint checkpoints/best.pth --image test.png --stage 2

# Inspect dataset tensors and save a debug figure
python dataset.py data/train
```

### Minimal Mamba-3 Reference

The repository also keeps the original minimal sequence-model reference:

```bash
python demo.py
python mamba3.py
```

## Data Format

Supported directory layouts:

```text
data/
  train/
    images/  *.jpg | *.png | *.jpeg | *.bmp | *.tiff
    labels/  *.json
```

Or place image files and matching JSON files in the same directory.

Supported JSON variants:

- LabelMe `shapes` with `shape_type` in `linestrip`, `line`, `polyline`, or `lines`
- Custom `{"curves": [...]}`
- Custom `{"annotations": [...]}`

Example:

```json
{
  "shapes": [
    {
      "shape_type": "linestrip",
      "points": [[x0, y0], [x1, y1], [x2, y2]],
      "label": "curve_1"
    }
  ]
}
```

The dataset pipeline rasterizes polyline annotations into supervision tensors including:

- `curve_mask`
- `centerline_mask`
- `instance_ids`
- `direction_vectors`
- `crossing_mask`
- `boundary_mask`
- `layering_target`
- `grid_mask`

Notes:

- `grid_mask` falls back to background derived from `instance_ids` when explicit grid labels are absent.
- `legend_patches` are not produced by the default dataset pipeline.
- `layering_target` is created as a placeholder and is only useful if you add real layering annotations.

## Default Configuration

### Training Defaults

| Item | Default |
| --- | --- |
| model | `sota` |
| image size | `512` |
| batch size | `4` |
| epochs | `100` |
| learning rate | `2e-4` |
| weight decay | `1e-4` |
| grad clip | `1.0` |
| warmup epochs | `5` |
| ema decay | `0.999` |
| loss ramp epochs | `10` |
| encoder dims | `64 / 128 / 256 / 512` |
| blocks per stage | `2 / 2 / 4 / 2` |
| decoder dim | `128` |
| num queries | `240` |
| query layers | `6` |
| query heads | `8` |
| align top-k | `96` |
| cross-attn top-k | `1024` |
| query routing | `True` |
| memory bottleneck ratio | `0.5` |

### Inference Defaults

| Item | Default |
| --- | --- |
| score threshold | `0.35` |
| mask threshold | `0.5` |
| top-k | `120` |
| min pixels | `24` |
| NMS IoU | `0.7` |
| crossing IoU override | `0.92` |
| crossing confidence threshold | `0.4` |
| crossing minimum overlap | `5` |
| quality power | `1.0` |

## Evaluation

Validation during training uses `evaluate.py` and reports metrics such as:

- `mAP50`, `mAP75`, `mAP50:95`
- `PQ`
- `skeleton_recall`
- `centerline_iou`, `centerline_dice`
- `curve_iou`, `curve_dice` for base-model composed masks

If `pycocotools` is installed, official COCO AP is computed through `COCOeval`. Otherwise the code falls back to a non-official interpolation path.

## Loss System

The SOTA criterion is organized as:

- Group A: `cls`, `mask`, `dice`, `quality`, `aux`, `dn_mask`, `otm`
- Group B: `centerline`, `crossing`, `boundary`, `direction`, `grid`
- Group C: `cape`, `pcc`
- Group D: `snake_offset`
- Group E: `legend_contrastive`
- Optional, disabled by default: `topograph`, `efd`

This means the current implementation uses 16 primary loss terms, with 2 extra optional terms available for experiments.

## Optional Modules and Caveats

- Legend-guided modules `A / LCAB / C / E` are implemented in code, but default training does not activate them because `legend_patches` are not provided by the standard dataloader.
- `use_style_head` and `use_layering_head` are disabled by default and should only be enabled when matching annotations exist.
- `crossing_logits` directly participates in inference-time suppression logic. Other topology heads mainly contribute during training.

## Tests

Run the segmentation-side tests with:

```bash
pytest tests/test_curve_model.py
pytest tests/test_curve_loss.py
pytest tests/test_curve_eval.py
pytest tests/test_legend.py
```

The repository also keeps the original Mamba-3 minimal reference tests:

```bash
pytest tests/test_mimo.py
pytest tests/test_parity.py
pytest tests/test_text.py
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `train.py` | training entry point for base and SOTA models |
| `curve_sota_query_seg.py` | default query-based segmentation model and criterion |
| `mamba3_curve_instance_seg.py` | base segmentation model and spatial encoder |
| `mamba3.py` | minimal Mamba-3 sequence-model implementation |
| `dataset.py` | polyline dataset parsing, rasterization, and augmentation |
| `evaluate.py` | validation metrics and COCO AP helpers |
| `profile_model.py` | profiling utility |
| `visualize_offsets.py` | snake offset visualization |
| `legend_encoder.py` | optional legend-guided query components |
| `tests/` | segmentation and minimal-model tests |
| `index.html` | English static method page |

## Documentation Page

`index.html` is ready to be published as a static webpage through GitHub Pages, Netlify, or Cloudflare Pages.

## Credits

- Albert Gu and Tri Dao for the Mamba architecture family
- `tommyip/mamba2-minimal` for the SSD chunking reference that inspired the pure PyTorch path
- `johnma2006/mamba-minimal` for the minimal and educational implementation style

## License

This project is released under the Apache 2.0 license. See `LICENSE`.
