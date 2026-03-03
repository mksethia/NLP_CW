#!/usr/bin/env python3
"""
Ensemble Voting Model — Don't Patronize Me!  (Training)

Fine-tune RoBERTa, ModernBERT, and MPNet for binary PCL classification.
Checkpoints are saved under BestModel/results/<ModelName>_final/.

Usage:
    python train.py                          # train all models
    python train.py --log-file /path/to.log  # custom log location
"""

import argparse
import gc
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
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


def setup_file_logging(log_file: Path, logger_name: str = "train"):
    """Route all stdout/stderr and logging output to a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger(logger_name)

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


# ── Training ──────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PCL ensemble models")
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(ROOT / "BestModel" / "logs" / "train.log"),
        help="Path to log file. All stdout/stderr is redirected here.",
    )
    args = parser.parse_args()

    setup_file_logging(Path(args.log_file))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([1.0, CLASS_WEIGHT_POS], dtype=torch.float32).to(device)
    print(f"Device: {device}  |  Class weights: {class_weights.tolist()}")

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_pcl_dataframe()
    train_ds, val_ds, _test_ds = make_splits(df)
    print(f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(_test_ds)} (held out for eval.py)")

    # ── Tokenisers ────────────────────────────────────────────────────────
    tokenisers = {
        name: AutoTokenizer.from_pretrained(path)
        for name, path in MODEL_CATALOGUE.items()
    }

    # ── Train each model ──────────────────────────────────────────────────
    for name, path in MODEL_CATALOGUE.items():
        tok_train = tokenize(train_ds, tokenisers[name])
        tok_val   = tokenize(val_ds,   tokenisers[name])

        train_model(name, path, tokenisers[name], tok_train, tok_val, class_weights)
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll models trained. Run eval.py to evaluate.")


if __name__ == "__main__":
    main()
