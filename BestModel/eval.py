#!/usr/bin/env python3
"""
Ensemble Voting Model — Don't Patronize Me!  (Evaluation)

Load trained RoBERTa, ModernBERT, and MPNet checkpoints, evaluate
per-model and ensemble performance, and write a Markdown report.

Usage:
    python eval.py                              # evaluate and write report
    python eval.py --report eval_report.md      # custom report filename
    python eval.py --log-file /path/to/eval.log # custom log location
"""

import argparse
import gc
import glob
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

# Import shared utilities from train.py
from train import (
    ROOT,
    DATA_DIR,
    RESULTS_DIR,
    MODEL_CATALOGUE,
    HYPERPARAMS,
    MAX_LENGTH,
    NUM_LABELS,
    LABEL_NAMES,
    CLASS_WEIGHT_POS,
    ENSEMBLE_WEIGHTS,
    ENSEMBLE_THRESHOLD,
    setup_file_logging,
    clean_text,
    load_pcl_dataframe,
    make_splits,
    WeightedTrainer,
    compute_metrics,
    tokenize,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# ── Model Loading ─────────────────────────────────────────────────────────────

def reload_model(name, tokenizer, val_ds, class_weights):
    """Reload the best checkpoint for a model."""
    output_dir = str(RESULTS_DIR / f"{name}_final")
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint under {output_dir}. Run train.py first.")
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


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_probs(trainer, tokenizer, dataset):
    """Run prediction → (hard labels, PCL probabilities)."""
    tok_ds = tokenize(dataset, tokenizer)
    out = trainer.predict(tok_ds)
    logits = out.predictions
    probs  = scipy_softmax(logits, axis=-1)
    preds  = np.argmax(logits, axis=-1)
    return preds, probs[:, 1]


# ── Visualisation ─────────────────────────────────────────────────────────────

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


# ── Ensemble Logic ────────────────────────────────────────────────────────────

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


# ── Report Writer ─────────────────────────────────────────────────────────────

def write_report(
    report_path: Path,
    per_model_reports: dict,
    per_model_metrics: dict,
    ensemble_report: str,
    ensemble_metrics: dict,
    summary_df: pd.DataFrame,
    n_misclassified: int,
    total_test: int,
):
    """Write a Markdown evaluation report to disk."""
    lines = [
        f"# Evaluation Report — Don't Patronize Me!",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Models | {', '.join(MODEL_CATALOGUE.keys())} |",
        f"| Ensemble weights | {ENSEMBLE_WEIGHTS} |",
        f"| Ensemble threshold | {ENSEMBLE_THRESHOLD} |",
        f"| Class weight (PCL) | {CLASS_WEIGHT_POS} |",
        f"| Max sequence length | {MAX_LENGTH} |",
        f"| Test samples | {total_test} |",
        f"",
        f"---",
        f"",
    ]

    # Per-model sections
    for name in MODEL_CATALOGUE:
        m = per_model_metrics[name]
        lines += [
            f"## {name}",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy  | {m['accuracy']:.4f} |",
            f"| Precision | {m['precision']:.4f} |",
            f"| Recall    | {m['recall']:.4f} |",
            f"| F1        | {m['f1']:.4f} |",
            f"",
            f"<details><summary>Full classification report</summary>",
            f"",
            f"```",
            per_model_reports[name],
            f"```",
            f"</details>",
            f"",
            f"Confusion matrix: `figures/{name}_cm.png`",
            f"",
        ]

    # Ensemble section
    lines += [
        f"---",
        f"",
        f"## Ensemble (Weighted Soft Vote)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy  | {ensemble_metrics['accuracy']:.4f} |",
        f"| Precision | {ensemble_metrics['precision']:.4f} |",
        f"| Recall    | {ensemble_metrics['recall']:.4f} |",
        f"| F1        | {ensemble_metrics['f1']:.4f} |",
        f"",
        f"<details><summary>Full classification report</summary>",
        f"",
        f"```",
        ensemble_report,
        f"```",
        f"</details>",
        f"",
        f"Confusion matrix: `figures/ensemble_cm.png`",
        f"",
    ]

    # Summary table
    lines += [
        f"---",
        f"",
        f"## Summary",
        f"",
        f"```",
        summary_df.to_string(float_format="{:.4f}".format),
        f"```",
        f"",
        f"Misclassified samples: **{n_misclassified}** / {total_test} "
        f"({n_misclassified / total_test * 100:.1f}%)",
        f"",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PCL ensemble models")
    parser.add_argument(
        "--report",
        type=str,
        default="eval_report.md",
        help="Filename for the evaluation report (written to BestModel/).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(ROOT / "BestModel" / "logs" / "eval.log"),
        help="Path to log file. All stdout/stderr is redirected here.",
    )
    args = parser.parse_args()

    setup_file_logging(Path(args.log_file), logger_name="eval")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([1.0, CLASS_WEIGHT_POS], dtype=torch.float32).to(device)
    print(f"Device: {device}  |  Class weights: {class_weights.tolist()}")

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_pcl_dataframe()
    _train_ds, val_ds, test_ds = make_splits(df)
    print(f"Using test split: {len(test_ds)} samples")

    # ── Tokenisers ────────────────────────────────────────────────────────
    tokenisers = {
        name: AutoTokenizer.from_pretrained(path)
        for name, path in MODEL_CATALOGUE.items()
    }

    # ── Load trained checkpoints ──────────────────────────────────────────
    models, trainers = {}, {}
    for name in MODEL_CATALOGUE:
        tok_val = tokenize(val_ds, tokenisers[name])
        model, trainer = reload_model(name, tokenisers[name], tok_val, class_weights)
        models[name]   = model
        trainers[name] = trainer
        torch.cuda.empty_cache()
        gc.collect()

    # ── Per-model evaluation on internal test split ───────────────────────
    all_probs = {}
    true_labels = np.array(test_ds["label"])

    per_model_reports = {}
    per_model_metrics = {}

    for name in MODEL_CATALOGUE:
        preds, probs = predict_probs(trainers[name], tokenisers[name], test_ds)
        all_probs[name] = probs

        report_str = classification_report(
            true_labels, preds, target_names=LABEL_NAMES, digits=4,
        )
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels, preds, average="binary", pos_label=1, zero_division=0,
        )
        acc = accuracy_score(true_labels, preds)

        per_model_reports[name] = report_str
        per_model_metrics[name] = {
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        }

        print(f"\n{name}:")
        print(report_str)
        plot_confusion(true_labels, preds, name, FIGURES_DIR / f"{name}_cm.png")

    # ── Ensemble evaluation ───────────────────────────────────────────────
    ens_preds = ensemble_predict(all_probs)

    ens_report_str = classification_report(
        true_labels, ens_preds, target_names=LABEL_NAMES, digits=4,
    )
    ens_p, ens_r, ens_f1, _ = precision_recall_fscore_support(
        true_labels, ens_preds, average="binary", pos_label=1, zero_division=0,
    )
    ens_acc = accuracy_score(true_labels, ens_preds)
    ensemble_metrics = {
        "accuracy": ens_acc, "precision": ens_p, "recall": ens_r, "f1": ens_f1,
    }

    print("\nENSEMBLE (weighted soft vote):")
    print(ens_report_str)
    plot_confusion(true_labels, ens_preds, "Ensemble", FIGURES_DIR / "ensemble_cm.png")

    # ── Summary table ─────────────────────────────────────────────────────
    rows = []
    for name in MODEL_CATALOGUE:
        m = per_model_metrics[name]
        rows.append({
            "Model": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
        })
    rows.append({
        "Model": "Ensemble",
        "Precision": ens_p,
        "Recall": ens_r,
        "F1": ens_f1,
    })
    summary_df = pd.DataFrame(rows).set_index("Model")
    print("\nSummary:")
    print(summary_df.to_string(float_format="{:.4f}".format))

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

    # ── Write Markdown report ─────────────────────────────────────────────
    report_path = Path(__file__).resolve().parent / args.report
    write_report(
        report_path=report_path,
        per_model_reports=per_model_reports,
        per_model_metrics=per_model_metrics,
        ensemble_report=ens_report_str,
        ensemble_metrics=ensemble_metrics,
        summary_df=summary_df,
        n_misclassified=len(mis_rows),
        total_test=len(test_ds),
    )

    # ── Official predictions ──────────────────────────────────────────────
    print("\nGenerating official dev/test predictions...")
    generate_submissions(trainers, tokenisers)
    print("Done.")


if __name__ == "__main__":
    main()
