"""
CS 898BA - Image Analysis and Computer Vision
Extra Credit: Advanced Segmentation

Chosen architecture: SegFormer (Hierarchical Transformer-based Semantic
Segmentation), using the ADE20K-finetuned checkpoint from NVIDIA.

The same "unidentified figure" image from Homeworks 1 and 2 is pushed through
three input preparation channels and segmented by the transformer each time:

  Channel A (Baseline)                 original unmodified RGB
  Channel B (Perceptual Transform)     V channel of HSV histogram-equalized
  Channel C (Statistical Contrast)     each RGB channel equalized independently

For every channel the pipeline saves a multi-coloured semantic overlay with the
target figure outlined, and scores the predicted figure mask against the
GrabCut pseudo-ground-truth built in Homework 2 using IoU and Dice.

Usage:
    python advanced_segmentation.py \
        --image alien_image.png \
        --ground-truth results_hw2/ground_truth.png

Outputs land in ./results_ec/ , including a ready-to-paste README section.

Author: Abdul Aleem Mohammed
"""

import argparse
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

SEED = 898
RESULTS = "results_ec"
DEFAULT_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Part 2 - the three input preparation channels
# ---------------------------------------------------------------------------
def channel_a(img_bgr):
    """Baseline: the original, unmodified RGB image."""
    return img_bgr.copy()


def channel_b(img_bgr):
    """Perceptual transformation: equalize the V channel in HSV, back to BGR."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_eq = cv2.merge([h, s, cv2.equalizeHist(v)])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def channel_c(img_bgr):
    """Statistical contrast normalization: equalize each RGB channel alone."""
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


# ---------------------------------------------------------------------------
# SegFormer inference
# ---------------------------------------------------------------------------
def load_model(model_name):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded {model_name} on {device} "
          f"({len(model.config.id2label)} semantic classes)")
    return processor, model, device


def segment(img_bgr, processor, model, device):
    """Run SegFormer and return a full-resolution per-pixel class map."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits              # (1, C, H/4, W/4)
    h, w = img_bgr.shape[:2]
    upsampled = F.interpolate(logits, size=(h, w), mode="bilinear",
                              align_corners=False)
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def target_class_id(model):
    """ADE20K carries an explicit 'person' class - the label a human would
    assign to the figure. Returns None if the checkpoint has no such class."""
    for idx, name in model.config.id2label.items():
        if "person" in str(name).lower():
            return int(idx)
    return None


def best_overlap_class(pred_map, gt_mask):
    """Diagnostic: which predicted class actually covers the figure best?"""
    g = gt_mask > 0
    best, best_iou = None, 0.0
    for cid in np.unique(pred_map):
        p = pred_map == cid
        union = np.logical_or(p, g).sum()
        if not union:
            continue
        iou = np.logical_and(p, g).sum() / union
        if iou > best_iou:
            best, best_iou = int(cid), float(iou)
    return best, best_iou


# ---------------------------------------------------------------------------
# Part 3 - metrics and visualization
# ---------------------------------------------------------------------------
def iou_dice(pred_mask, gt_mask):
    p = pred_mask > 0
    g = gt_mask > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = float(inter / union) if union else 0.0
    denom = p.sum() + g.sum()
    dice = float(2 * inter / denom) if denom else 0.0
    return iou, dice


def ade_palette(n):
    """Deterministic multi-colour palette for the semantic overlay."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 255, size=(n, 3), dtype=np.uint8)


def make_overlay(img_bgr, pred_map, target_mask, palette, alpha=0.55):
    """Multi-coloured semantic overlay with the target figure outlined."""
    colour = palette[pred_map % len(palette)]                       # H x W x 3
    blend = cv2.addWeighted(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 1 - alpha,
                            colour, alpha, 0)
    if target_mask is not None and target_mask.any():
        contours, _ = cv2.findContours(target_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, contours, -1, (255, 255, 255), 3)
    return blend


def comparison_plot(inputs_bgr, overlays, gt_mask, metrics, model_name):
    """Original, both normalized inputs, all three segmentations, ground truth."""
    titles_in = ["Channel A - original RGB",
                 "Channel B - V-channel normalized",
                 "Channel C - per-channel normalized",
                 "Ground truth (HW2 GrabCut)"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for ax, key, title in zip(axes[0], ("A", "B", "C"), titles_in):
        ax.imshow(cv2.cvtColor(inputs_bgr[key], cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    axes[0, 3].imshow(gt_mask, cmap="gray")
    axes[0, 3].set_title(titles_in[3], fontsize=11)
    axes[0, 3].axis("off")

    for ax, key in zip(axes[1], ("A", "B", "C")):
        m = metrics[key]
        ax.imshow(overlays[key])
        ax.set_title(f"SegFormer on Channel {key}\n"
                     f"IoU={m['iou']:.4f}  Dice={m['dice']:.4f}", fontsize=11)
        ax.axis("off")

    ax = axes[1, 3]
    ax.axis("off")
    rows = "\n".join(
        f"  Channel {k}:   IoU {metrics[k]['iou']:.4f}    Dice {metrics[k]['dice']:.4f}"
        for k in ("A", "B", "C"))
    ax.text(0.0, 0.5,
            f"Model: SegFormer\n{model_name}\n\nFigure mask vs HW2 ground truth\n\n{rows}",
            fontsize=12, family="monospace", va="center")

    fig.suptitle("Extra Credit - SegFormer across three input preparation channels",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "advanced_comparison.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


def write_readme_section(metrics, model_name, target_name, diagnostics, hw2):
    """Emit a ready-to-append README section with the real numbers filled in."""
    best = max(metrics, key=lambda k: metrics[k]["iou"])
    rows = "\n".join(
        f"| Channel {k} — {d} | {metrics[k]['iou']:.4f} | {metrics[k]['dice']:.4f} |"
        for k, d in (("A", "original RGB"),
                     ("B", "V-channel normalized"),
                     ("C", "per-channel normalized")))
    hw2_rows = "\n".join(f"| {k} (HW2) | {v[0]:.4f} | {v[1]:.4f} |"
                         for k, v in hw2.items())
    diag = "\n".join(
        f"- Channel {k}: best-overlapping predicted class was "
        f"`{v['best_class_name']}` at IoU {v['best_class_iou']:.4f}"
        for k, v in diagnostics.items())

    text = f"""
---

# Extra Credit — Advanced Segmentation (SegFormer)

Branch: `Advanced-Segmentation`. The same unidentified-figure image from
Homeworks 1 and 2 is segmented by a hierarchical transformer instead of the
classical methods, across three input preparation channels.

## Chosen architecture

**SegFormer** — `{model_name}`, the ADE20K-finetuned checkpoint. SegFormer pairs
a hierarchical Mix Transformer encoder (multi-scale features, no positional
encodings) with an all-MLP decoder, so it reasons about global scene context
rather than the purely local intensity statistics the HW2 methods relied on.
The figure is isolated by taking the `{target_name}` class from the 150-class
ADE20K label space, which is the category a human annotator would assign.

## Run

```bash
pip install -r requirements.txt
python advanced_segmentation.py --image alien_image.png \\
    --ground-truth results_hw2/ground_truth.png
```

## Part 2 — the three input channels

| Channel | Preparation |
| --- | --- |
| A (Baseline) | original, unmodified RGB |
| B (Perceptual transformation) | V channel of HSV histogram-equalized, converted back to RGB |
| C (Statistical contrast normalization) | each RGB channel histogram-equalized independently |

## Part 3 — results

![SegFormer comparison across input channels](results_ec/advanced_comparison.png)

Predicted figure mask scored against the Homework 2 GrabCut pseudo-ground-truth:

| Input | IoU | Dice |
| --- | --- | --- |
{rows}

For reference, the classical Homework 2 results on the same ground truth:

| Method | IoU | Dice |
| --- | --- | --- |
{hw2_rows}

## Part 4 — analysis

**Against Homework 2.** The best transformer channel reached IoU
{metrics[best]['iou']:.4f} / Dice {metrics[best]['dice']:.4f}, compared with
0.1032 / 0.1871 for the strongest classical method in Homework 2 (HSV K-Means).
The classical methods only ever thresholded or clustered pixel values, so on a
dark frame where the figure's tones overlap the grass they inevitably dragged in
large regions of background. SegFormer instead assigns a semantic label per
pixel using global context, so it can separate a person-shaped region from lawn
and rooflines even when their raw intensities are close.

**Effect of the input channel.** {"Contrast normalization helped" if max(metrics['B']['iou'], metrics['C']['iou']) > metrics['A']['iou'] else "Contrast normalization did not help"}
here. The three channels are the same scene with different tone curves, and the
transformer was trained on ordinary photographs, so an input whose statistics
look like a normally exposed photo sits closer to its training distribution.
Aggressive per-channel equalization (Channel C) also breaks colour constancy,
which shifts the image further from what the encoder expects.

**Diagnostics.**

{diag}

**Limits of the ground truth.** The reference mask is itself a GrabCut estimate
from Homework 2 rather than hand-drawn truth, so these numbers measure agreement
with that estimate, not absolute accuracy. Where the transformer and GrabCut
disagree at the figure's boundary, it is not automatically the transformer that
is wrong.

## Extra credit files

```
advanced_segmentation.py          # full pipeline
results_ec/
├── advanced_comparison.png       # the required comparison plot
├── overlay_channel_[ABC].png     # multi-coloured semantic overlays
├── mask_channel_[ABC].png        # binary figure masks
├── metrics.txt                   # IoU / Dice table
└── metrics.json                  # machine-readable results
```
"""
    with open(os.path.join(RESULTS, "README_section.md"), "w") as f:
        f.write(text.lstrip("\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="alien_image.png")
    ap.add_argument("--ground-truth", default="results_hw2/ground_truth.png")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    for p in (args.image, args.ground_truth):
        if not os.path.exists(p):
            raise SystemExit(f"Required file not found: {p}")

    os.makedirs(RESULTS, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    gt = cv2.imread(args.ground_truth, cv2.IMREAD_GRAYSCALE)
    if gt.shape != img.shape[:2]:
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    print(f"image {img.shape} | ground truth covers "
          f"{100 * (gt > 0).mean():.1f}% of the frame")

    processor, model, device = load_model(args.model)
    palette = ade_palette(len(model.config.id2label))
    tgt = target_class_id(model)
    target_name = model.config.id2label[tgt] if tgt is not None else "n/a"
    print(f"target class: {target_name} (id {tgt})")

    inputs_bgr = {"A": channel_a(img), "B": channel_b(img), "C": channel_c(img)}
    overlays, metrics, diagnostics = {}, {}, {}

    for key in ("A", "B", "C"):
        src = inputs_bgr[key]
        cv2.imwrite(os.path.join(RESULTS, f"input_channel_{key}.png"), src)
        pred = segment(src, processor, model, device)

        mask = (pred == tgt).astype(np.uint8) * 255 if tgt is not None \
            else np.zeros(pred.shape, np.uint8)
        iou, dice = iou_dice(mask, gt)
        metrics[key] = {"iou": iou, "dice": dice,
                        "pixels": int((mask > 0).sum())}

        bc, bi = best_overlap_class(pred, gt)
        diagnostics[key] = {
            "best_class_id": bc,
            "best_class_name": model.config.id2label.get(bc, "n/a") if bc is not None else "n/a",
            "best_class_iou": bi,
        }

        overlays[key] = make_overlay(src, pred, mask > 0, palette)
        cv2.imwrite(os.path.join(RESULTS, f"mask_channel_{key}.png"), mask)
        cv2.imwrite(os.path.join(RESULTS, f"overlay_channel_{key}.png"),
                    cv2.cvtColor(overlays[key], cv2.COLOR_RGB2BGR))
        print(f"Channel {key}: IoU={iou:.4f}  Dice={dice:.4f}  "
              f"(figure pixels {metrics[key]['pixels']}, "
              f"best-overlap class '{diagnostics[key]['best_class_name']}' "
              f"@ IoU {bi:.4f})", flush=True)

    hw2 = {"Otsu": (0.0952, 0.1739), "Adaptive": (0.0961, 0.1754),
           "K-Means": (0.1032, 0.1871)}

    lines = ["=== Extra Credit - SegFormer figure mask vs HW2 ground truth ===",
             f"model: {args.model}", f"target class: {target_name}", "",
             f"{'Input':<34}{'IoU':>10}{'Dice':>10}", "-" * 54]
    for k, d in (("A", "original RGB"), ("B", "V-channel normalized"),
                 ("C", "per-channel normalized")):
        lines.append(f"{'Channel ' + k + ' - ' + d:<34}"
                     f"{metrics[k]['iou']:>10.4f}{metrics[k]['dice']:>10.4f}")
    lines += ["", "Classical Homework 2 baselines on the same ground truth:"]
    for k, (i, d) in hw2.items():
        lines.append(f"{k + ' (HW2)':<34}{i:>10.4f}{d:>10.4f}")
    report = "\n".join(lines)
    print("\n" + report, flush=True)

    with open(os.path.join(RESULTS, "metrics.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"model": args.model, "target_class": target_name,
                   "metrics": metrics, "diagnostics": diagnostics,
                   "hw2_baselines": hw2}, f, indent=2)

    comparison_plot(inputs_bgr, overlays, gt, metrics, args.model)
    write_readme_section(metrics, args.model, target_name, diagnostics, hw2)
    print(f"\nDone. See ./{RESULTS}/ "
          f"(README_section.md is ready to append to README.md).")


if __name__ == "__main__":
    main()
