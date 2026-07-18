---

# Extra Credit — Advanced Segmentation (SegFormer)

Branch: `Advanced-Segmentation`. The same unidentified-figure image from
Homeworks 1 and 2 is segmented by a hierarchical transformer instead of the
classical methods, across three input preparation channels.

## Chosen architecture

**SegFormer** — `nvidia/segformer-b4-finetuned-ade-512-512`, the ADE20K-finetuned checkpoint. SegFormer pairs
a hierarchical Mix Transformer encoder (multi-scale features, no positional
encodings) with an all-MLP decoder, so it reasons about global scene context
rather than the purely local intensity statistics the HW2 methods relied on.
The figure is isolated by taking the `person` class from the 150-class
ADE20K label space, which is the category a human annotator would assign.

## Run

```bash
pip install -r requirements.txt
python advanced_segmentation.py --image alien_image.png \
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
| Channel A — original RGB | 0.4267 | 0.5981 |
| Channel B — V-channel normalized | 0.4311 | 0.6025 |
| Channel C — per-channel normalized | 0.4353 | 0.6065 |

For reference, the classical Homework 2 results on the same ground truth:

| Method | IoU | Dice |
| --- | --- | --- |
| Otsu (HW2) | 0.0952 | 0.1739 |
| Adaptive (HW2) | 0.0961 | 0.1754 |
| K-Means (HW2) | 0.1032 | 0.1871 |

## Part 4 — analysis

**Against Homework 2.** The best transformer channel reached IoU
0.4353 / Dice 0.6065, compared with
0.1032 / 0.1871 for the strongest classical method in Homework 2 (HSV K-Means).
The classical methods only ever thresholded or clustered pixel values, so on a
dark frame where the figure's tones overlap the grass they inevitably dragged in
large regions of background. SegFormer instead assigns a semantic label per
pixel using global context, so it can separate a person-shaped region from lawn
and rooflines even when their raw intensities are close.

**Effect of the input channel.** Contrast normalization helped
here. The three channels are the same scene with different tone curves, and the
transformer was trained on ordinary photographs, so an input whose statistics
look like a normally exposed photo sits closer to its training distribution.
Aggressive per-channel equalization (Channel C) also breaks colour constancy,
which shifts the image further from what the encoder expects.

**Diagnostics.**

- Channel A: best-overlapping predicted class was `person` at IoU 0.4267
- Channel B: best-overlapping predicted class was `person` at IoU 0.4311
- Channel C: best-overlapping predicted class was `person` at IoU 0.4353

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
