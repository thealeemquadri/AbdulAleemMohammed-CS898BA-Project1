# Abdul Aleem Mohammed-CS898BA-Project1

**CS 898BA – Image Analysis and Computer Vision — Homework One**
Author: Abdul Aleem Mohammed

A single-script OpenCV pipeline that analyzes the low-light "alien" doorbell
capture, builds a full set of color-space / transformed / blurred variants, and
runs four edge-detection techniques over a random subset. The whole assignment
runs from one file: `main.py`.

\---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python hello\_world.py                       # prints "Hello World!"
python main.py --input alien\_image.png      # runs the full pipeline
```

Generated images go to `output/` (git-ignored — regenerate by re-running). The
six README plots and the statistics report are written to the committed
`results/` folder. A fixed seed (`SEED = 898`) makes every run reproducible.

\---

## Image-count checkpoints

The script prints a running count and a final summary so the spec's checkpoints
are verifiable. All counts are confirmed on disk:

|Stage|Required|Produced|
|-|-|-|
|Base images (original, greyscale, binary, HSV, CIELAB, HLS, V-equalized→RGB)|7|7|
|After 2 unique affine transforms per image|21|21|
|After 7-level Gaussian blur|168|168|
|Chosen subset (168 ÷ 4)|42|42|
|After Sobel / Laplacian / Canny / Prewitt (42 inputs + 168 results)|210|210|
|Five-image comparison plots|42|42|

\---

## Part 2 — Analysis and generation

### 2.1 Per-channel statistics

Computed with NumPy and `scipy.stats` and written to `results/statistics.txt`:

```
Channel    Min   Max     Mean  Median  Mode     Skew  Range      Std        Var
-------------------------------------------------------------------------------
Blue         0   255    20.87    10.0     3    1.690    255   25.975     674.73
Green        0   255    23.69    15.0    11    1.778    255   21.924     480.67
Red          0   255    19.64    11.0     2    2.137    255   22.158     490.99
```

The low means (\~20/255) and strong positive skew (1.7–2.1) quantify what the eye
sees: this is a very dark, low-key image where most pixels sit near black, with a
long tail of brighter pixels from the porch lights and sky. This directly
motivates the V-channel histogram equalization in step 2.3.

### 2.2–2.5 Color spaces and lighting normalization

Greyscale, binary (threshold at 127), HSV, CIELAB, and HLS are generated and
saved. Histogram equalization is then applied to the **V** channel of the HSV
image (`cv2.equalizeHist`) to redistribute the compressed dark tones, and the
result is converted back to RGB. On this image the normalization noticeably
lifts the figure and houses out of the shadows. → **7 base images.**

### 2.6 Affine transforms

Each base image receives **2 unique** affine transforms, drawn from a fixed pool
of 14 transforms that are each distinct in type or value (rotations of 15–270°,
three translations, three scales, two shears). No two of the 14 are identical.
→ **21 images.**

### 2.7–2.8 Gaussian blur and the effect of σ

Every one of the 21 images is blurred at σ = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5.
As σ grows the kernel widens and averages over a larger neighborhood: at σ ≈ 0.5
only fine sensor noise is suppressed while edges stay crisp; by σ ≈ 1.5–2.5
small features and texture begin to dissolve; by σ ≈ 3.0–3.5 the figure smears
into the background. For a noisy night capture, a light blur (σ ≈ 1.0) is the
sweet spot — it removes speckle without destroying the silhouette we need for
detection. → **168 images total.**

\---

## Part 3 — Edge detection

The 168 images are shuffled and split into 4 equal subsets of 42; the first is
chosen. Sobel, Laplacian, Canny, and Prewitt are applied to each, and both the
input (before) and all four results (after) are saved → **210 images**. A
five-panel comparison plot is produced for each of the 42 subset images; six are
exported below.

### Sample comparison plots

Each plot's title shows the exact processing chain of that subset image
(e.g. `greyscale\_rotate\_90\_blur2.5`).

!\[Sample plot 1](results/plot\_05.png)
!\[Sample plot 2](results/plot\_20.png)
!\[Sample plot 3](results/plot\_26.png)
!\[Sample plot 4](results/plot\_33.png)
!\[Sample plot 5](results/plot\_39.png)
!\[Sample plot 6](results/plot\_40.png)

### Comparison and recommendation

|Technique|Pros|Cons|
|-|-|-|
|**Sobel**|Cheap, gives gradient magnitude + direction, robust on smooth gradients|Thick edges, no thinning, somewhat noise-sensitive|
|**Laplacian**|Single-pass, omnidirectional second derivative|Very noise-sensitive (amplifies high frequencies), double edges|
|**Canny**|Multi-stage (smoothing → gradient → non-max suppression → hysteresis), thin clean edges when tuned|Threshold-dependent; with fixed high thresholds it drops weak edges entirely|
|**Prewitt**|Simple, uniform-weight gradient, similar coverage to Sobel|Noisier than Sobel, thick edges, no post-processing|

**Best technique for this image set: Sobel (with Prewitt close behind).** This is
the interesting result and it runs against the usual "Canny is best" reflex. On
this specific low-light, low-contrast set, **Canny produces almost nothing** — its
default 100/200 hysteresis thresholds are far too high for an image whose
gradients are tiny (means near 20/255), and the subset's blurred members weaken
those gradients further. Sobel and Prewitt, which output raw gradient magnitude
with no hysteresis gate, recover the figure's outline and the rooflines far more
completely. Laplacian picks up the shape too but is the noisiest of the four.

If Canny were required to win here, it would need its thresholds dropped
dramatically (e.g. \~20/60) and/or the V-equalized image as input rather than the
dark original — a good follow-up experiment.

\---

## Repository layout

```
.
├── main.py                 # full Part 2 + Part 3 pipeline (single script)
├── hello\_world.py          # initial-commit script
├── requirements.txt
├── .gitignore
├── AI\_Log.md               # AI usage log

├── README.md
├── alien\_image.png         # the assignment image
├── images/
│   ├── base/               # the 7 base images (committed)
│   ├── affine/             # the 14 affine transforms (committed)
│   └── plots/              # all 42 comparison plots (committed)
├── results/                # 6 README plots + statistics.txt (committed)
└── output/                 # 147 blurred + 210 edge images (git-ignored, regenerated on run)
```

