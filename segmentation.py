"""
CS 898BA - Image Analysis and Computer Vision
Homework Two: Image Segmentation

Pipeline:
  Part 2 - multi-channel normalization: histogram-equalize all 3 channels
           independently, merge -> fully normalized color image.
  Part 3 - threshold segmentation: Otsu global + adaptive Gaussian, with
           binary masks and foreground extractions for each.
  Part 4 - K-Means colour clustering in HSV (optimal K chosen from 3-5),
           isolating the cluster that captures the figure.
  Part 5 - evaluation: a GrabCut pseudo-ground-truth mask, IoU (Jaccard) and
           Dice for all 3 methods, and a side-by-side comparison plot.

Usage:
    python segmentation.py --input alien_image.png

Outputs land in ./segmentation_output/ ; the comparison plot, normalized image,
and metrics are also copied to ./results_hw2/ (committed) for the README.

Author: Abdul Aleem Mohammed
"""

import argparse
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

SEED = 898
np.random.seed(SEED)

OUT = "segmentation_output"
RESULTS = "results_hw2"

# Approximate bounding box around the walking figure (x, y, w, h) on the
# 1017x555 image, used to seed the GrabCut pseudo-ground-truth and to pick the
# K-Means cluster that best captures the figure.
FIGURE_BBOX = (495, 85, 165, 425)


def ensure_dirs():
    for d in (OUT, RESULTS):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Part 2 - multi-channel normalization
# ---------------------------------------------------------------------------
def normalize_multichannel(img):
    """Histogram-equalize each BGR channel independently, then merge."""
    b, g, r = cv2.split(img)
    eq = cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])
    cv2.imwrite(os.path.join(OUT, "normalized_color.png"), eq)
    cv2.imwrite(os.path.join(RESULTS, "normalized_color.png"), eq)
    print("Part 2 - saved fully normalized color image.")
    return eq


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def extract_foreground(color_img, mask):
    """Keep original colour where mask==255, black elsewhere."""
    return cv2.bitwise_and(color_img, color_img, mask=mask)


def orient_to_figure(mask, bbox):
    """Ensure the figure region is the white (255) class.

    Polarity of a threshold is just a convention, so we pick the orientation
    whose white pixels are denser inside the figure box than outside.
    """
    x, y, w, h = bbox
    inside = mask[y:y + h, x:x + w]
    total_in = inside.size
    total_out = mask.size - total_in
    white_in = np.count_nonzero(inside)
    white_out = np.count_nonzero(mask) - white_in
    dens_in = white_in / max(total_in, 1)
    dens_out = white_out / max(total_out, 1)
    if dens_in < dens_out:           # figure is currently the black class -> flip
        return cv2.bitwise_not(mask)
    return mask


# ---------------------------------------------------------------------------
# Part 3 - threshold segmentation
# ---------------------------------------------------------------------------
def threshold_segmentation(norm):
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT, "normalized_gray.png"), gray)

    # Otsu global threshold.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = orient_to_figure(otsu, FIGURE_BBOX)

    # Adaptive Gaussian threshold.
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    adaptive = orient_to_figure(adaptive, FIGURE_BBOX)

    for name, mask in (("otsu", otsu), ("adaptive", adaptive)):
        cv2.imwrite(os.path.join(OUT, f"mask_{name}.png"), mask)
        cv2.imwrite(os.path.join(OUT, f"foreground_{name}.png"),
                    extract_foreground(norm, mask))
    print("Part 3 - saved Otsu and adaptive masks + foreground extractions.")
    return otsu, adaptive


# ---------------------------------------------------------------------------
# Part 4 - K-Means colour clustering
# ---------------------------------------------------------------------------
def kmeans_segmentation(norm):
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Choose optimal K in [3, 5] by silhouette score on a random subsample.
    sample_idx = np.random.choice(len(pixels), size=4000, replace=False)
    sample = pixels[sample_idx]
    best_k, best_score, fitted = 3, -1.0, {}
    for k in (3, 4, 5):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels_full = km.fit_predict(pixels)
        fitted[k] = (km, labels_full)
        score = silhouette_score(sample, km.predict(sample))
        print(f"  K={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"Part 4 - optimal K = {best_k} (silhouette={best_score:.4f})")

    km, labels = fitted[best_k]
    labels_img = labels.reshape(hsv.shape[:2])

    # Save the full clustered visualisation (each cluster -> its mean colour).
    centers = km.cluster_centers_.astype(np.uint8)
    clustered = centers[labels].reshape(hsv.shape)
    cv2.imwrite(os.path.join(OUT, "kmeans_clusters_hsv.png"),
                cv2.cvtColor(clustered, cv2.COLOR_HSV2BGR))

    # Isolate the cluster densest inside the figure bbox (vs the rest of frame).
    x, y, w, h = FIGURE_BBOX
    best_cluster, best_ratio = 0, -1.0
    for c in range(best_k):
        cmask = (labels_img == c)
        inside = cmask[y:y + h, x:x + w].sum()
        outside = cmask.sum() - inside
        ratio = inside / (outside + 1e-6)
        if ratio > best_ratio:
            best_cluster, best_ratio = c, ratio

    kmeans_mask = np.where(labels_img == best_cluster, 255, 0).astype(np.uint8)
    kmeans_mask = orient_to_figure(kmeans_mask, FIGURE_BBOX)
    cv2.imwrite(os.path.join(OUT, "mask_kmeans.png"), kmeans_mask)
    cv2.imwrite(os.path.join(OUT, "foreground_kmeans.png"),
                extract_foreground(norm, kmeans_mask))
    print("Part 4 - saved K-Means mask + foreground extraction.")
    return kmeans_mask, best_k


# ---------------------------------------------------------------------------
# Part 5 - evaluation
# ---------------------------------------------------------------------------
def build_ground_truth(norm):
    """Pseudo-ground-truth figure mask via seeded GrabCut on the normalized image.

    GrabCut on the raw dark frame collapses to background, so we run it on the
    higher-contrast normalized image and seed it explicitly: a vertical core of
    the figure as definite foreground and a wide border as definite background.
    """
    x, y, w, h = FIGURE_BBOX
    mask = np.full(norm.shape[:2], cv2.GC_PR_BGD, np.uint8)
    mask[y:y + h, x:x + w] = cv2.GC_PR_FGD
    cx = x + w // 2
    mask[y + 30:y + h - 20, cx - 25:cx + 25] = cv2.GC_FGD       # definite figure core
    mask[:, :max(x - 40, 0)] = cv2.GC_BGD                        # definite background
    mask[:, x + w + 40:] = cv2.GC_BGD
    mask[:max(y - 40, 0), :] = cv2.GC_BGD
    mask[y + h + 30:, :] = cv2.GC_BGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(norm, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    gt = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    gt = cv2.morphologyEx(gt, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gt = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cv2.imwrite(os.path.join(OUT, "ground_truth.png"), gt)
    cv2.imwrite(os.path.join(RESULTS, "ground_truth.png"), gt)
    print("Part 5 - built pseudo-ground-truth (seeded GrabCut on normalized image).")
    return gt


def iou_dice(pred, gt):
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 0.0
    dice = (2 * inter) / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0
    return iou, dice


def evaluate(masks, gt):
    lines = ["=== Part 5 - Quantitative segmentation metrics (vs GrabCut pseudo-GT) ===",
             f"{'Method':<12}{'IoU (Jaccard)':>16}{'Dice':>10}", "-" * 38]
    metrics = {}
    for name, m in masks.items():
        iou, dice = iou_dice(m, gt)
        metrics[name] = (iou, dice)
        lines.append(f"{name:<12}{iou:>16.4f}{dice:>10.4f}")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    with open(os.path.join(RESULTS, "metrics.txt"), "w") as f:
        f.write(report + "\n")
    return metrics


def comparison_plot(original, norm, masks, gt, metrics):
    panels = [
        ("Original (HW1)", cv2.cvtColor(original, cv2.COLOR_BGR2RGB), None, None),
        ("Multi-channel normalized", cv2.cvtColor(norm, cv2.COLOR_BGR2RGB), None, None),
        ("Pseudo ground truth", gt, "gray", None),
        ("Otsu mask", masks["otsu"], "gray", "otsu"),
        ("Adaptive mask", masks["adaptive"], "gray", "adaptive"),
        ("K-Means mask", masks["kmeans"], "gray", "kmeans"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, im, cmap, key) in zip(axes.flat, panels):
        ax.imshow(im, cmap=cmap)
        sub = ""
        if key in metrics:
            iou, dice = metrics[key]
            sub = f"\nIoU={iou:.3f}  Dice={dice:.3f}"
        ax.set_title(title + sub, fontsize=11)
        ax.axis("off")
    fig.suptitle("HW2 Image Segmentation - method comparison", fontsize=14)
    fig.tight_layout()
    for d in (OUT, RESULTS):
        fig.savefig(os.path.join(d, "comparison_plot.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("Part 5 - saved comparison plot.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="alien_image.png")
    args = ap.parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"Input '{args.input}' not found.")

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    ensure_dirs()

    norm = normalize_multichannel(img)                 # Part 2
    otsu, adaptive = threshold_segmentation(norm)       # Part 3
    kmeans_mask, best_k = kmeans_segmentation(norm)     # Part 4

    masks = {"otsu": otsu, "adaptive": adaptive, "kmeans": kmeans_mask}
    gt = build_ground_truth(norm)                       # Part 5
    metrics = evaluate(masks, gt)
    comparison_plot(img, norm, masks, gt, metrics)

    print("\nDone. See ./segmentation_output/ (full) and ./results_hw2/ (README assets).")


if __name__ == "__main__":
    main()
