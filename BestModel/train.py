#!/usr/bin/env python3
"""
Ensemble Voting Model — Don't Patronize Me!

Binary PCL classification using RoBERTa, ModernBERT, and MPNet
with weighted soft-vote ensemble.

Usage:
    python train.py              # train all models from scratch
    python train.py --reload     # reload from saved checkpoints
    python train.py --log-file /path/to/train.log
"""

import argparse
import gc
import glob
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "Dont_Patronize_Me_Trainingset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_CATALOGUE = {
    "RoBERTa":    "FacebookAI/roberta-base",
    "ModernBERT": "answerdotai/ModernBERT-base",
    "MPNet":      "microsoft/mpnet-base",
}

HYPERPARAMS = {
    "learning_rate":    2e-5,
    "num_train_epochs": 4,
    "batch_size":       32,
    "weight_decay":     0.06,
}

MAX_LENGTH       = 128
NUM_LABELS       = 2
LABEL_NAMES      = ["Non-PCL", "PCL"]
CLASS_WEIGHT_POS = 9.0

ENSEMBLE_WEIGHTS   = {"RoBERTa": 1.2, "ModernBERT": 0.9, "MPNet": 0.9}
ENSEMBLE_THRESHOLD = 1.38


# ── File Logging ──────────────────────────────────────────────────────────────

class StreamToLogger:
    """Redirect writes from stdout/stderr into the Python logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        if not message:
            return
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            line = self._buffer.rstrip()
            if line:
                self.logger.log(self.level, line)
            self._buffer = ""


def setup_file_logging(log_file: Path):
    """Route all stdout/stderr and logging output to a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger("train")

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    print("=" * 80)
    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Logging to: {log_file}")
    print("=" * 80)


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip HTML tags, HTML entities, and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)              # HTML tags
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)   # HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_pcl_dataframe() -> pd.DataFrame:
    """Load dontpatronizeme_pcl.tsv with binary labels and cleaned text."""
    cols = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df = pd.read_csv(
        DATA_DIR / "dontpatronizeme_pcl.tsv",
        sep="\t", skiprows=4, names=cols,
        on_bad_lines="skip", engine="python",
    )
    df = df.dropna(subset=["text", "label"])
    df["par_id"] = df["par_id"].astype(int)
    df["label"]  = df["label"].astype(int)
    df["binary_label"] = (df["label"] >= 2).astype(int)
    df["text"] = df["text"].astype(str).apply(clean_text)

    n_pcl  = (df["binary_label"] == 1).sum()
    n_npcl = (df["binary_label"] == 0).sum()
    print(f"Loaded {len(df)} samples  |  Non-PCL: {n_npcl}  PCL: {n_pcl}  "
          f"(imbalance {n_npcl / n_pcl:.1f}:1)")
    return df


def make_splits(df: pd.DataFrame):
    """80/10/10 stratified split → HuggingFace Datasets."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["binary_label"], random_state=42,
    )
    to_hf = lambda d: Dataset.from_dict({
        "text":  d["text"].tolist(),
        "label": d["binary_label"].tolist(),
    })
    return to_hf(train_df), to_hf(val_df), to_hf(test_df)


# ── Trainer with Class Weights ────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """CrossEntropyLoss with class weights to handle imbalance."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        w = self._class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0,
    )
    return {"accuracy": accuracy_score(labels, preds),
            "precision": p, "recall": r, "f1": f1}


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )


# ── Training / Reload ─────────────────────────────────────────────────────────

def train_model(name, model_path, tokenizer, train_ds, val_ds, class_weights):
    """Fine-tune a single transformer from scratch."""
    output_dir = str(RESULTS_DIR / f"{name}_final")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=HYPERPARAMS["num_train_epochs"],
        per_device_train_batch_size=HYPERPARAMS["batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["batch_size"],
        learning_rate=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        seed=42,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    print(f"  {name}: training complete")
    return model, trainer


def reload_model(name, tokenizer, val_ds, class_weights):
    """Reload the best checkpoint for a model."""
    output_dir = str(RESULTS_DIR / f"{name}_final")
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint under {output_dir}. Train first.")
    ckpt = ckpts[-1]
    print(f"  {name}: reloading from {ckpt}")

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=NUM_LABELS,
    )
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=HYPERPARAMS["batch_size"],
        report_to="none",
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    return model, trainer


# ── Evaluation Helpers ────────────────────────────────────────────────────────

def predict_probs(trainer, tokenizer, dataset):
    """Run prediction → (hard labels, PCL probabilities)."""
    tok_ds = tokenize(dataset, tokenizer)
    out = trainer.predict(tok_ds)
    logits = out.predictions
    probs  = scipy_softmax(logits, axis=-1)
    preds  = np.argmax(logits, axis=-1)
    return preds, probs[:, 1]


def plot_confusion(true, pred, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cm = confusion_matrix(true, pred)
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=axes[0], cmap="Blues", colorbar=False,
    )
    axes[0].set_title(f"{title} — Counts")
    cm_n = confusion_matrix(true, pred, normalize="true")
    ConfusionMatrixDisplay(cm_n, display_labels=LABEL_NAMES).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2%",
    )
    axes[1].set_title(f"{title} — Normalised")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def ensemble_predict(model_probs: dict) -> np.ndarray:
    """Weighted soft vote over model PCL probabilities."""
    n = len(next(iter(model_probs.values())))
    wsum = np.zeros(n)
    for name, probs in model_probs.items():
        wsum += ENSEMBLE_WEIGHTS[name] * probs
    k = len(model_probs)
    return (wsum / k > ENSEMBLE_THRESHOLD / k).astype(int)


# ── Official Submission ───────────────────────────────────────────────────────

def generate_submissions(trainers, tokenisers):
    """Produce dev.txt and test.txt for the shared-task leaderboard."""

    # Official dev set — map par_ids to cleaned text
    dev_labels_df = pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")
    full_df = load_pcl_dataframe()
    pid2text = dict(zip(full_df["par_id"], full_df["text"]))
    dev_texts = [pid2text[int(pid)] for pid in dev_labels_df["par_id"]]
    dev_hf = Dataset.from_dict({"text": dev_texts})

    # Official test set
    test_df = pd.read_csv(
        DATA_DIR / "task4_test.tsv", sep="\t", header=None,
        names=["id", "art_id", "keyword", "country_code", "text"],
    )
    test_df["text"] = test_df["text"].astype(str).apply(clean_text)
    test_hf = Dataset.from_dict({"text": test_df["text"].tolist()})

    for label, hf_ds, fname in [
        ("dev",  dev_hf,  ROOT / "dev.txt"),
        ("test", test_hf, ROOT / "test.txt"),
    ]:
        probs = {}
        for name in MODEL_CATALOGUE:
            _, p = predict_probs(trainers[name], tokenisers[name], hf_ds)
            probs[name] = p
        preds = ensemble_predict(probs)
        with open(fname, "w") as f:
            for v in preds:
                f.write(f"{v}\n")
        print(f"  {label}.txt: {len(preds)} predictions ({preds.sum()} PCL) -> {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate PCL ensemble")
    parser.add_argument("--reload", action="store_true",
                        help="Skip training, load saved checkpoints")
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(ROOT / "BestModel" / "logs" / "train.log"),
        help="Path to log file. All stdout/stderr is redirected here.",
    )
    args = parser.parse_args()

    setup_file_logging(Path(args.log_file))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([1.0, CLASS_WEIGHT_POS], dtype=torch.float32).to(device)
    print(f"Device: {device}  |  Class weights: {class_weights.tolist()}")

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_pcl_dataframe()
    train_ds, val_ds, test_ds = make_splits(df)
    print(f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── Tokenisers ────────────────────────────────────────────────────────
    tokenisers = {
        name: AutoTokenizer.from_pretrained(path)
        for name, path in MODEL_CATALOGUE.items()
    }

    # ── Train / Reload each model ─────────────────────────────────────────
    models, trainers = {}, {}
    for name, path in MODEL_CATALOGUE.items():
        tok_train = tokenize(train_ds, tokenisers[name])
        tok_val   = tokenize(val_ds,   tokenisers[name])

        if args.reload:
            model, trainer = reload_model(name, tokenisers[name], tok_val, class_weights)
        else:
            model, trainer = train_model(
                name, path, tokenisers[name], tok_train, tok_val, class_weights,
            )

        models[name]   = model
        trainers[name] = trainer
        torch.cuda.empty_cache()
        gc.collect()

    # ── Per-model evaluation on internal test split ───────────────────────
    all_probs = {}
    true_labels = np.array(test_ds["label"])

    for name in MODEL_CATALOGUE:
        preds, probs = predict_probs(trainers[name], tokenisers[name], test_ds)
        all_probs[name] = probs
        print(f"\n{name}:")
        print(classification_report(true_labels, preds,
                                    target_names=LABEL_NAMES, digits=4))
        plot_confusion(true_labels, preds, name,
                       FIGURES_DIR / f"{name}_cm.png")

    # ── Ensemble evaluation ───────────────────────────────────────────────
    ens_preds = ensemble_predict(all_probs)
    print("\nENSEMBLE (weighted soft vote):")
    print(classification_report(true_labels, ens_preds,
                                target_names=LABEL_NAMES, digits=4))
    plot_confusion(true_labels, ens_preds, "Ensemble",
                   FIGURES_DIR / "ensemble_cm.png")

    # Summary table
    rows = []
    for name in MODEL_CATALOGUE:
        p_arr = (all_probs[name] > 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels, p_arr, average="binary", pos_label=1,
        )
        rows.append({"Model": name, "Precision": p, "Recall": r, "F1": f1})
    p, r, f1, _ = precision_recall_fscore_support(
        true_labels, ens_preds, average="binary", pos_label=1,
    )
    rows.append({"Model": "Ensemble", "Precision": p, "Recall": r, "F1": f1})
    print("\nSummary:")
    print(pd.DataFrame(rows).set_index("Model").to_string(float_format="{:.4f}".format))

    # ── Misclassification log ─────────────────────────────────────────────
    mis_path = ROOT / "ensemble_misclassifications.csv"
    mis_rows = []
    for i in range(len(test_ds)):
        if ens_preds[i] != true_labels[i]:
            mis_rows.append({
                "text": test_ds[i]["text"],
                "true_label": int(true_labels[i]),
                "predicted_label": int(ens_preds[i]),
            })
    pd.DataFrame(mis_rows).to_csv(mis_path, index=False)
    print(f"Misclassifications: {len(mis_rows)} -> {mis_path}")

    # ── Official predictions ──────────────────────────────────────────────
    print("\nGenerating official dev/test predictions...")
    generate_submissions(trainers, tokenisers)
    print("Done.")


if __name__ == "__main__":
    main()
