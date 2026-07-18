"""
CS 898BA - Image Analysis and Computer Vision
Homework Three: Deep Learning for Fish Classification

Pipeline:
  Part 2 - data pipeline: stratified 70/15/15 split, resize to 128x128,
           normalize to [0, 1], and augment the training set only
           (random horizontal flip, minor rotation, brightness jitter).
  Part 3 - baseline CNN built from scratch in Keras: 4 convolutional blocks
           (32 -> 64 -> 128 -> 128) with ReLU + MaxPooling, then Flatten,
           a fully connected hidden layer, Dropout, and a softmax output.
           Trained with Adam, lr = 0.001, batch size 32, fixed epochs.
  Part 4 - hyperparameter optimization by seeded Random Search over a
           12-point grid: learning rate {0.01, 0.001, 0.0001} x
           batch size {32, 64} x dropout {0.3, 0.5}. Best trial is selected
           on validation loss and retrained to full length.
  Part 5 - evaluation of baseline vs optimized on the held-out test set:
           accuracy / precision / recall / F1, per-class classification
           reports, loss+accuracy curves, and a confusion matrix.

Usage:
    python classification.py --data Fish
    python classification.py --data Fish --smoke      # fast sanity run

Outputs:
    models/        trained weights (baseline + optimized)
    results_hw3/   plots, metrics, classification reports, search table

Author: Abdul Aleem Mohammed
"""

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 898
IMG_SIZE = 128
MODELS_DIR = "models"
RESULTS_DIR = "results_hw3"
CACHE = f"cache_{IMG_SIZE}.npz"

keras.utils.set_random_seed(SEED)


def ensure_dirs():
    for d in (MODELS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Part 2 - data loading, stratified splitting, augmentation
# ---------------------------------------------------------------------------
def load_dataset(data_dir):
    """Decode every image once to a uint8 array and cache it to disk."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], [str(c) for c in d["classes"]]

    classes = sorted(
        c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))
    )
    X, y = [], []
    for ci, c in enumerate(classes):
        folder = os.path.join(data_dir, c)
        for fn in sorted(os.listdir(folder)):
            path = os.path.join(folder, fn)
            try:
                im = Image.open(path).convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR
                )
            except Exception:
                continue  # skip unreadable files
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(ci)
    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    np.savez_compressed(CACHE, X=X, y=y, classes=np.array(classes))
    return X, y, classes


def stratified_split(X, y):
    """70 / 15 / 15 stratified train / validation / test split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def build_augmenter():
    """Training-time augmentation: flips, minor rotations, brightness jitter."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.06, fill_mode="reflect", seed=SEED),
            layers.RandomBrightness(0.15, value_range=(0.0, 1.0), seed=SEED),
        ],
        name="augmentation",
    )


def make_dataset(X, y, batch_size, training, augmenter=None):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda a, b: (tf.cast(a, tf.float32) / 255.0, b),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training and augmenter is not None:
        ds = ds.map(
            lambda a, b: (tf.clip_by_value(augmenter(a, training=True), 0.0, 1.0), b),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def save_augmentation_samples(X_tr, classes, y_tr):
    """Visual evidence that augmentation is doing something sensible."""
    aug = build_augmenter()
    idx = np.arange(4)
    base = tf.cast(X_tr[idx], tf.float32) / 255.0
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for r in range(4):
        axes[r, 0].imshow(base[r].numpy())
        axes[r, 0].set_title(f"original ({classes[y_tr[idx[r]]]})", fontsize=9)
        axes[r, 0].axis("off")
        for c in range(1, 5):
            out = tf.clip_by_value(aug(base[r: r + 1], training=True), 0.0, 1.0)
            axes[r, c].imshow(out[0].numpy())
            axes[r, c].set_title(f"augmented {c}", fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("Part 2 - training-set augmentation samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "augmentation_samples.png"), dpi=100,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 3 - baseline CNN architecture
# ---------------------------------------------------------------------------
def build_model(num_classes, dropout=0.3, lr=1e-3):
    """Custom CNN: 4 conv blocks with increasing filters, then dense head."""
    model = keras.Sequential(
        [
            layers.Input((IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="fish_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model, train_ds, val_ds, epochs, early_stop=False):
    """Fit a model. The optimized run additionally uses a learning-rate
    schedule and a longer early-stopping patience: with a fixed learning rate
    the validation loss is noisy, so a short patience stops while validation
    accuracy is still improving (this is exactly what happened on the first
    attempt). ReduceLROnPlateau smooths the late epochs and lets the model
    converge instead of being truncated."""
    cbs = []
    if early_stop:
        cbs += [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True, verbose=0
            ),
        ]
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2
    )
    return hist.history


# ---------------------------------------------------------------------------
# Part 4 - hyperparameter optimization (seeded random search)
# ---------------------------------------------------------------------------
def random_search(splits, classes, n_trials, epochs):
    (X_tr, y_tr), (X_val, y_val), _ = splits
    grid = [
        {"lr": lr, "batch": b, "dropout": d}
        for lr in (0.01, 0.001, 0.0001)
        for b in (32, 64)
        for d in (0.3, 0.5)
    ]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(grid))[:n_trials]
    trials = []
    aug = build_augmenter()

    for i, gi in enumerate(order, 1):
        cfg = grid[gi]
        print(f"\n--- trial {i}/{len(order)}: {cfg} ---", flush=True)
        keras.utils.set_random_seed(SEED)
        tr = make_dataset(X_tr, y_tr, cfg["batch"], True, aug)
        va = make_dataset(X_val, y_val, cfg["batch"], False)
        model = build_model(len(classes), dropout=cfg["dropout"], lr=cfg["lr"])
        h = train(model, tr, va, epochs)
        rec = dict(cfg)
        rec["val_loss"] = float(np.min(h["val_loss"]))
        rec["val_accuracy"] = float(np.max(h["val_accuracy"]))
        trials.append(rec)
        print(f"    val_loss={rec['val_loss']:.4f}  val_acc={rec['val_accuracy']:.4f}",
              flush=True)
        keras.backend.clear_session()

    trials.sort(key=lambda r: r["val_loss"])
    best = trials[0]

    lines = ["=== Part 4 - Random Search over the 12-point hyperparameter grid ===",
             f"(sampled {len(order)} of {len(grid)} configurations, {epochs} epochs each, seed={SEED})",
             "",
             f"{'rank':<6}{'lr':>9}{'batch':>7}{'dropout':>9}{'val_loss':>11}{'val_acc':>10}",
             "-" * 52]
    for r, t in enumerate(trials, 1):
        lines.append(f"{r:<6}{t['lr']:>9}{t['batch']:>7}{t['dropout']:>9}"
                     f"{t['val_loss']:>11.4f}{t['val_accuracy']:>10.4f}")
    covered_lr = sorted({t["lr"] for t in trials})
    covered_b = sorted({t["batch"] for t in trials})
    covered_d = sorted({t["dropout"] for t in trials})
    lines += ["",
              f"coverage -> learning rates: {covered_lr}",
              f"            batch sizes:    {covered_b}",
              f"            dropout rates:  {covered_d}",
              "",
              f"BEST (lowest validation loss): lr={best['lr']}, batch={best['batch']}, "
              f"dropout={best['dropout']}  (val_loss={best['val_loss']:.4f})"]
    report = "\n".join(lines)
    print("\n" + report + "\n", flush=True)
    with open(os.path.join(RESULTS_DIR, "hyperparameter_search.txt"), "w") as f:
        f.write(report + "\n")
    return best, trials


# ---------------------------------------------------------------------------
# Part 5 - evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_te, y_te, classes, name, batch=32):
    ds = make_dataset(X_te, y_te, batch, False)
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(axis=1)
    acc = accuracy_score(y_te, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro",
                                                  zero_division=0)
    rep = classification_report(y_te, pred, target_names=classes, digits=4,
                                zero_division=0)
    text = (f"=== {name} - test-set classification report ===\n\n{rep}\n"
            f"Accuracy : {acc:.4f}\nPrecision (macro): {p:.4f}\n"
            f"Recall    (macro): {r:.4f}\nF1-score  (macro): {f1:.4f}\n")
    print(text, flush=True)
    with open(os.path.join(RESULTS_DIR,
                           f"classification_report_{name.lower()}.txt"), "w") as f:
        f.write(text)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "pred": pred.tolist()}


def visualization(hist_b, hist_o, cm, classes, mb, mo):
    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])

    def curves(ax, h, key, title):
        ax.plot(h[key], label=f"train {key}")
        ax.plot(h[f"val_{key}"], label=f"val {key}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    curves(fig.add_subplot(gs[0, 0]), hist_b, "loss", "Baseline - loss")
    curves(fig.add_subplot(gs[0, 1]), hist_b, "accuracy", "Baseline - accuracy")
    curves(fig.add_subplot(gs[1, 0]), hist_o, "loss", "Optimized - loss")
    curves(fig.add_subplot(gs[1, 1]), hist_o, "accuracy", "Optimized - accuracy")

    ax = fig.add_subplot(gs[:, 2])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Optimized model - confusion matrix (test set)", fontsize=11)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"HW3 Fish Classification - baseline (test acc {mb['accuracy']:.3f}) vs "
        f"optimized (test acc {mo['accuracy']:.3f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_comparison.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Fish")
    ap.add_argument("--epochs-baseline", type=int, default=25)
    ap.add_argument("--epochs-search", type=int, default=10)
    ap.add_argument("--epochs-final", type=int, default=40)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="fast sanity run")
    ap.add_argument(
        "--stage", choices=["all", "optimized"], default="all",
        help="'optimized' reuses the saved baseline model and search results and "
             "only retrains the optimized model, then re-evaluates both.",
    )
    args = ap.parse_args()

    if args.smoke:
        args.epochs_baseline = args.epochs_search = args.epochs_final = 2
        args.trials = 2

    ensure_dirs()
    t0 = time.time()

    # ---- Part 2 -----------------------------------------------------------
    X, y, classes = load_dataset(args.data)
    splits = stratified_split(X, y)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = splits
    print(f"Part 2 - classes: {classes}")
    print(f"Part 2 - split sizes  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")
    print(f"Part 2 - train per-class: {np.bincount(y_tr).tolist()}")
    print(f"Part 2 - test  per-class: {np.bincount(y_te).tolist()}", flush=True)
    save_augmentation_samples(X_tr, classes, y_tr)

    aug = build_augmenter()

    if args.stage == "optimized":
        # Reuse the already-trained baseline and the completed search.
        prev = json.load(open(os.path.join(RESULTS_DIR, "history.json")))
        hist_b = prev["baseline"]
        best = {k: prev["best_config"][k] for k in ("lr", "batch", "dropout")}
        trials = prev["trials"]
        baseline = keras.models.load_model(
            os.path.join(MODELS_DIR, "baseline_model.keras")
        )
        print(f"Reusing saved baseline and search results; best config = {best}",
              flush=True)
    else:
        # ---- Part 3 - baseline --------------------------------------------
        print("\n=== Part 3 - training baseline (Adam, lr=0.001, batch=32, "
              "dropout=0.3) ===", flush=True)
        keras.utils.set_random_seed(SEED)
        tr32 = make_dataset(X_tr, y_tr, 32, True, aug)
        va32 = make_dataset(X_val, y_val, 32, False)
        baseline = build_model(len(classes), dropout=0.3, lr=1e-3)
        baseline.summary()
        hist_b = train(baseline, tr32, va32, args.epochs_baseline)
        baseline.save(os.path.join(MODELS_DIR, "baseline_model.keras"))

        # ---- Part 4 - random search ---------------------------------------
        best, trials = random_search(splits, classes, args.trials, args.epochs_search)

    # ---- retrain best config to full length -------------------------------
    print(f"\n=== Part 4 - retraining best config {best} for {args.epochs_final} epochs ===",
          flush=True)
    keras.utils.set_random_seed(SEED)
    tr_b = make_dataset(X_tr, y_tr, best["batch"], True, aug)
    va_b = make_dataset(X_val, y_val, best["batch"], False)
    optimized = build_model(len(classes), dropout=best["dropout"], lr=best["lr"])
    hist_o = train(optimized, tr_b, va_b, args.epochs_final, early_stop=True)
    optimized.save(os.path.join(MODELS_DIR, "optimized_model.keras"))

    # ---- Part 5 - evaluation ----------------------------------------------
    print("\n=== Part 5 - test-set evaluation ===", flush=True)
    mb = evaluate(baseline, X_te, y_te, classes, "Baseline")
    mo = evaluate(optimized, X_te, y_te, classes, "Optimized", batch=best["batch"])
    cm = confusion_matrix(y_te, np.array(mo["pred"]))
    np.savetxt(os.path.join(RESULTS_DIR, "confusion_matrix_optimized.csv"), cm,
               fmt="%d", delimiter=",")

    summary = [
        "=== Part 5 - Baseline vs Optimized (held-out test set) ===",
        "",
        f"{'Model':<12}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}",
        "-" * 51,
        f"{'Baseline':<12}{mb['accuracy']:>10.4f}{mb['precision']:>11.4f}"
        f"{mb['recall']:>9.4f}{mb['f1']:>9.4f}",
        f"{'Optimized':<12}{mo['accuracy']:>10.4f}{mo['precision']:>11.4f}"
        f"{mo['recall']:>9.4f}{mo['f1']:>9.4f}",
        "",
        f"Baseline config : lr=0.001, batch=32, dropout=0.3 "
        f"({args.epochs_baseline} epochs, fixed)",
        f"Optimized config: lr={best['lr']}, batch={best['batch']}, "
        f"dropout={best['dropout']} (up to {args.epochs_final} epochs, early stopping)",
        "",
        "Precision / recall / F1 are macro-averaged across the "
        f"{len(classes)} fish classes.",
    ]
    text = "\n".join(summary)
    print("\n" + text, flush=True)
    with open(os.path.join(RESULTS_DIR, "metrics_summary.txt"), "w") as f:
        f.write(text + "\n")

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({"baseline": hist_b, "optimized": hist_o, "best_config": best,
                   "trials": trials,
                   "test_metrics": {"baseline": {k: mb[k] for k in
                                                 ("accuracy", "precision", "recall", "f1")},
                                    "optimized": {k: mo[k] for k in
                                                  ("accuracy", "precision", "recall", "f1")}}},
                  f, indent=2)

    visualization(hist_b, hist_o, cm, classes, mb, mo)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. "
          f"See ./{RESULTS_DIR}/ and ./{MODELS_DIR}/.", flush=True)


if __name__ == "__main__":
    main()
