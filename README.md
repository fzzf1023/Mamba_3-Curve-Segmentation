<div align="center">
  <h1>Mamba-3 Chart Curve Instance Segmentation</h1>
  <p><strong>Query-based chart curve instance segmentation built on a Mamba-3 spatial encoder.</strong></p>
  <p>Pure PyTorch training and inference for thin-curve extraction, crossings, topology-aware supervision, and static web publishing.</p>
  <p>
    <a href="#installation"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="#model-variants"><img src="https://img.shields.io/badge/default-CurveSOTAQueryNet-0969da?style=for-the-badge" alt="Default Model"></a>
    <a href="#default-configuration"><img src="https://img.shields.io/badge/input-RGB%20%2B%20HSV%20%2B%20Sobel-1f6feb?style=for-the-badge" alt="Input Stack"></a>
    <a href="#default-configuration"><img src="https://img.shields.io/badge/inference-crossing--aware%20NMS-1f883d?style=for-the-badge" alt="Inference"></a>
    <a href="index.html"><img src="https://img.shields.io/badge/docs-index.html-8250df?style=for-the-badge" alt="Docs"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-57606a?style=for-the-badge" alt="License"></a>
  </p>
  <p>
    <a href="#overview"><img src="https://img.shields.io/badge/Overview-open-24292f?style=flat-square" alt="Overview"></a>
    <a href="#architecture"><img src="https://img.shields.io/badge/Architecture-open-24292f?style=flat-square" alt="Architecture"></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-open-24292f?style=flat-square" alt="Quick Start"></a>
    <a href="#default-configuration"><img src="https://img.shields.io/badge/Defaults-open-24292f?style=flat-square" alt="Defaults"></a>
    <a href="#documentation-page"><img src="https://img.shields.io/badge/Static_Docs-open-24292f?style=flat-square" alt="Static Docs"></a>
  </p>
</div>

> [!IMPORTANT]
> This README is aligned with the current local implementation. The default training entry point is `CurveSOTAQueryNet` via `train.py --model sota`.
>
> The segmentation stack is the primary project in this repository. The standalone `mamba3.py` reference remains available as a compact implementation of the sequence-model core.

## Overview

This repository combines two related parts:

- A chart curve instance segmentation system with a Mamba-3 spatial encoder
- A minimal `mamba3.py` sequence-model implementation used by the spatial backbone and kept as a standalone reference

The segmentation side is the primary project in the current codebase. It includes `CurveInstanceMamba3Net`, the default `CurveSOTAQueryNet`, and an English `index.html` method page that can be published directly as a static site.

## Project Snapshot

<table>
  <tr>
    <td valign="top" width="33%">
      <strong>Default Entry</strong><br>
      <code>train.py --model sota</code><br><br>
      Query-based instance segmentation with Hungarian matching, one-to-many supervision, and denoising queries.
    </td>
    <td valign="top" width="33%">
      <strong>Backbone</strong><br>
      <code>3 / 5 / 9 / 9</code> progressive branches<br><br>
      Mamba spatial encoder with local, directional, reverse, and snake scan paths over <code>H / V / D45 / D135</code>.
    </td>
    <td valign="top" width="33%">
      <strong>Decoder</strong><br>
      Query decoder + topology heads<br><br>
      FPN fusion, <code>H/2</code> stem skip, and additive grid suppression feed parallel instance and topology prediction heads.
    </td>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <strong>Supervision</strong><br>
      <code>16</code> primary losses in <code>5</code> groups<br><br>
      Instance, topology, consistency, snake-offset, and optional legend-guided supervision are organized into one criterion.
    </td>
    <td valign="top" width="33%">
      <strong>Inference</strong><br>
      Quality-aware scoring + crossing-aware NMS<br><br>
      The default post-process path preserves genuine crossings while removing duplicates and small fragments.
    </td>
    <td valign="top" width="33%">
      <strong>Publishing</strong><br>
      <a href="index.html"><code>index.html</code></a><br><br>
      The repository already contains an English static method page suitable for GitHub Pages, Netlify, or Cloudflare Pages.
    </td>
  </tr>
</table>

## Quick Navigation

<table>
  <tr>
    <td valign="top" width="33%">
      <a href="#architecture"><strong>Architecture</strong></a><br>
      Backbone, decoder, topology heads, and the train/infer split.
    </td>
    <td valign="top" width="33%">
      <a href="#quick-start"><strong>Quick Start</strong></a><br>
      Training, resume, ablation, profiling, and visualization commands.
    </td>
    <td valign="top" width="33%">
      <a href="#default-configuration"><strong>Default Configuration</strong></a><br>
      Training defaults, decoder settings, and inference thresholds.
    </td>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <a href="#data-format"><strong>Data Format</strong></a><br>
      Supported JSON schemas and generated supervision tensors.
    </td>
    <td valign="top" width="33%">
      <a href="#loss-system"><strong>Loss System</strong></a><br>
      The current 16-term default objective and optional extras.
    </td>
    <td valign="top" width="33%">
      <a href="#documentation-page"><strong>Documentation Page</strong></a><br>
      Static publishing notes for the bundled English method page.
    </td>
  </tr>
</table>

## Highlights

| Area | Current implementation | Why it matters |
| --- | --- | --- |
| Input | RGB + HSV + Sobel enhancement is enabled by default in the SOTA model. | Improves thin-curve separation under low contrast and cluttered backgrounds. |
| Backbone | Progressive branch schedules `3 / 5 / 9 / 9` over `H/4 -> H/32`. | Balances high-resolution detail with stronger long-range structure modeling. |
| Branch types | Local depthwise convolution, row/column Mamba, reverse row/column scans, and snake scans in `H / V / D45 / D135`. | Covers both directional continuity and curved geometry. |
| Decoder | FPN fusion, `H/2` stem skip, additive grid-suppression bias, query decoder, and topology heads. | Preserves details while separating instance decoding from topology supervision. |
| Training | Hungarian matching, one-to-many matching, denoising queries, EMA, and loss-ramp scheduling. | Improves optimization stability and supervision density. |
| Losses | 16 primary terms across 5 groups, plus `topograph` and `efd` as default-disabled optional terms. | Keeps the default path strong while leaving room for controlled experiments. |
| Inference | Quality-aware scoring, crossing-aware NMS, and fragment filtering. | Improves crossing recall without letting duplicates dominate the output. |
| Legend path | Implemented in code, but inactive by default because the standard dataset pipeline does not emit `legend_patches`. | Keeps the repository extensible without overstating default behavior. |

## Architecture

<table>
  <tr>
    <td valign="top" width="25%">
      <strong>01. Input Stack</strong><br>
      RGB + HSV + Sobel enhancement<br><br>
      The default SOTA path augments raw RGB before the first convolutional stem.
    </td>
    <td valign="top" width="25%">
      <strong>02. Progressive Encoder</strong><br>
      <code>H/4 -> H/32</code> with <code>3 / 5 / 9 / 9</code> branches<br><br>
      Spatial Mamba blocks mix local depthwise paths with directional and snake scans for curved structures.
    </td>
    <td valign="top" width="25%">
      <strong>03. Dual Prediction Heads</strong><br>
      Query decoder || topology heads<br><br>
      Fused features feed instance-level queries and dense topology supervision in parallel.
    </td>
    <td valign="top" width="25%">
      <strong>04. Split Endpoints</strong><br>
      Train branch || infer branch<br><br>
      Training aggregates matching and multi-group losses; inference keeps the query path dominant and uses crossing logits for post-processing.
    </td>
  </tr>
</table>

### Reference Flow

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

The repository keeps two segmentation variants that share the same Mamba-style spatial backbone but target different use cases.

| Variant | Class | Recommended use | What it predicts |
| --- | --- | --- | --- |
| Base | `CurveInstanceMamba3Net` | Baseline experiments, lighter ablations, dense pixel prediction | A composed curve mask, instance embeddings, and topology-related dense heads |
| SOTA | `CurveSOTAQueryNet` | Default training, instance-level evaluation, best overall performance | Query logits, per-instance masks, mask quality, and auxiliary topology heads |

In short:

- `Base` is a dense prediction baseline centered on `centerline + width -> composed_mask`.
- `SOTA` is the main query-based instance segmentation model with matching, denoising, and crossing-aware post-processing.

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

<details>
<summary><strong>Minimal Mamba-3 Reference</strong></summary>

<br>

The repository also keeps the original minimal sequence-model reference:

```bash
python demo.py
python mamba3.py
```

</details>

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

Validation during training uses `evaluate.py` with an extraction-first metric design.

Primary metrics for the repository's final goal, curve segmentation and extraction:

- `curve_iou`, `curve_dice`: overlap quality of the final extracted curve mask
- `curve_precision`, `curve_recall`: how cleanly the extracted curve set covers GT curves
- `curve_cldice`: topology-aware quality of the extracted curves
- `skeleton_recall`: how much of the GT centerline is covered by the final extracted result

Auxiliary dense-head diagnostics:

- `centerline_iou`, `centerline_dice`
- `centerline_precision`, `centerline_recall`

Instance-level metrics for the default `CurveSOTAQueryNet` only:

- `mAP50`, `mAP75`, and `mAP50:95`
- `PQ`
- Optional official COCO AP via `pycocotools`: `coco_mAP50`, `coco_mAP75`, and `coco_mAP50:95`

Important detail:

- For `CurveSOTAQueryNet`, instance metrics and merged curve-mask metrics are computed from the post-processed inference results, not from raw decoder queries.
- For the base model, `curve_*` metrics are computed from the dense `composed_mask`.
- `COCO AP` is treated as a supplemental instance-separation metric, not the primary metric for curve extraction quality.

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

<details>
<summary><strong>Run segmentation-side tests</strong></summary>

<br>

```bash
pytest tests/test_curve_model.py
pytest tests/test_curve_loss.py
pytest tests/test_curve_eval.py
pytest tests/test_legend.py
```

</details>

<details>
<summary><strong>Run minimal Mamba-3 reference tests</strong></summary>

<br>

```bash
pytest tests/test_mimo.py
pytest tests/test_parity.py
pytest tests/test_text.py
```

</details>

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

> [!TIP]
> `index.html` is ready to be published as a static webpage through GitHub Pages, Netlify, or Cloudflare Pages.

## Credits

- Albert Gu and Tri Dao for the Mamba architecture family
- `tommyip/mamba2-minimal` for the SSD chunking reference that inspired the pure PyTorch path
- `johnma2006/mamba-minimal` for the minimal and educational implementation style

## License

This project is released under the Apache 2.0 license. See `LICENSE`.
