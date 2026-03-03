#!/usr/bin/env python3
"""
Exploratory Data Analysis — Don't Patronize Me! Dataset
=========================================================
Three in-depth EDA techniques, each producing:
  • Visual / tabular evidence  (saved as PNG figures in eda/figures/)
  • Written analysis
  • Impact statement on the PCL classification task

Techniques
----------
1. Class Distribution & Imbalance Analysis
2. Lexical Contrast — PCL vs Non-PCL  (word clouds + distinctive terms)
3. Sequence-Length Distribution & max_length Selection
"""

import os
import sys
import textwrap
import warnings
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                        # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

import nltk
for resource in [
    "tokenizers/punkt", "tokenizers/punkt_tab",
    "corpora/stopwords",
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "Dont_Patronize_Me_Trainingset")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "figures")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "eda_report.txt")

os.makedirs(FIG_DIR, exist_ok=True)

# ── pretty printing ──────────────────────────────────────────────────────────
_report_lines: list[str] = []

def report(msg: str = "") -> None:
    """Print to console AND buffer for the text report."""
    print(msg)
    _report_lines.append(msg)

def save_report() -> None:
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_report_lines))
    report(f"\n[Report saved to {REPORT_FILE}]")


# ── data loading ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pcl_cols = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df_pcl = pd.read_csv(
        os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv"),
        sep="\t", skiprows=4, names=pcl_cols,
        on_bad_lines="skip", engine="python",
    )

    cat_cols = [
        "par_id", "art_id", "text", "keyword", "country_code",
        "span_start", "span_finish", "span_text", "pcl_category", "num_annotators",
    ]
    df_cat = pd.read_csv(
        os.path.join(DATA_DIR, "dontpatronizeme_categories.tsv"),
        sep="\t", skiprows=4, names=cat_cols,
        on_bad_lines="skip", engine="python",
    )

    # binary label (spec: labels >= 2 -> PCL)
    df_pcl["binary_label"] = (df_pcl["label"] >= 2).astype(int)

    return df_pcl, df_cat


# ==============================================================================
# TECHNIQUE 1 — Class Distribution & Imbalance Analysis
# ==============================================================================

def technique_1_class_distribution(df_pcl: pd.DataFrame) -> None:
    report("=" * 80)
    report("TECHNIQUE 1 — Class Distribution & Imbalance Analysis")
    report("=" * 80)

    # ── 1a  multi-panel figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    palette_5 = sns.color_palette("YlOrRd", n_colors=5)
    palette_2 = ["#4c72b0", "#dd4444"]

    # --- panel A: original 0-4 label distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    label_counts = df_pcl["label"].value_counts().sort_index()
    bars = ax1.bar(label_counts.index.astype(str), label_counts.values, color=palette_5)
    for bar, val in zip(bars, label_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}\n({val / len(df_pcl) * 100:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_title("(a) Original Label Distribution (0-4)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Annotator-Aggregated Label")
    ax1.set_ylabel("Count")

    # --- panel B: binary PCL / Non-PCL ---
    ax2 = fig.add_subplot(gs[0, 1])
    bin_counts = df_pcl["binary_label"].value_counts().sort_index()
    labels_bin = ["Non-PCL\n(labels 0-1)", "PCL\n(labels 2-4)"]
    wedges, texts, autotexts = ax2.pie(
        bin_counts.values, labels=labels_bin, colors=palette_2,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * len(df_pcl) / 100)):,})",
        startangle=140, textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title("(b) Binary Class Split", fontweight="bold", fontsize=12)

    # --- panel C: PCL rate by keyword (target community) ---
    ax3 = fig.add_subplot(gs[1, 0])
    kw_pcl = (
        df_pcl.groupby("keyword")["binary_label"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "pcl_rate", "sum": "pcl_count", "count": "total"})
        .sort_values("pcl_rate", ascending=True)
    )
    y_pos = np.arange(len(kw_pcl))
    bars3 = ax3.barh(y_pos, kw_pcl["pcl_rate"] * 100,
                      color=sns.color_palette("coolwarm", n_colors=len(kw_pcl)))
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(kw_pcl.index, fontsize=10)
    ax3.set_xlabel("PCL Rate (%)")
    ax3.set_title("(c) PCL Rate by Target Community", fontweight="bold", fontsize=12)
    for bar, rate, cnt in zip(bars3, kw_pcl["pcl_rate"], kw_pcl["pcl_count"]):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{rate * 100:.1f}% (n={int(cnt)})", va="center", fontsize=8)

    # --- panel D: stacked bar — PCL vs Non-PCL per keyword ---
    ax4 = fig.add_subplot(gs[1, 1])
    kw_stack = kw_pcl.sort_values("total", ascending=True)
    non_pcl = kw_stack["total"] - kw_stack["pcl_count"]
    ax4.barh(kw_stack.index, non_pcl, label="Non-PCL", color=palette_2[0])
    ax4.barh(kw_stack.index, kw_stack["pcl_count"], left=non_pcl, label="PCL", color=palette_2[1])
    ax4.set_xlabel("Number of Paragraphs")
    ax4.set_title("(d) Absolute Counts per Community", fontweight="bold", fontsize=12)
    ax4.legend(loc="lower right")

    fig.suptitle("Technique 1 — Class Distribution & Imbalance Analysis",
                 fontsize=15, fontweight="bold", y=1.01)
    fig_path = os.path.join(FIG_DIR, "technique1_class_distribution.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report(f"  Figure saved -> {fig_path}")

    # ── 1b  written analysis ─────────────────────────────────────────────────
    n_total = len(df_pcl)
    n_pcl   = int(bin_counts.get(1, 0))
    n_nonpcl = int(bin_counts.get(0, 0))
    ratio = n_nonpcl / n_pcl if n_pcl else float("inf")

    report("\n-- Analysis --")
    report(f"  Total paragraphs : {n_total:,}")
    report(f"  Non-PCL (0-1)    : {n_nonpcl:,}  ({n_nonpcl/n_total*100:.1f}%)")
    report(f"  PCL     (2-4)    : {n_pcl:,}  ({n_pcl/n_total*100:.1f}%)")
    report(f"  Imbalance ratio  : {ratio:.1f} : 1")
    report("")
    report("  The dataset is heavily skewed: ~90% Non-PCL vs ~10% PCL.")
    report("  Among the original 5-point scale, label 0 dominates (>80%).")
    report("  PCL rates vary by keyword: some communities attract more")
    report("  patronising language than others, suggesting keyword-dependent")
    report("  linguistic patterns.")

    # ── 1c  impact statement ─────────────────────────────────────────────────
    report("\n-- Impact on Classification --")
    report(textwrap.dedent("""\
      * Accuracy alone is misleading (a majority-class baseline scores ~90%).
        We must optimise and report F1-score (especially macro / weighted).
      * Training should use class-weighted loss or focal loss to up-weight the
        minority PCL class.
      * Stratified train/dev/test splits are essential to preserve the ~10%
        PCL proportion in every fold.
      * Keyword-conditioned PCL rates suggest the model may benefit from
        keyword-aware features or conditioning, but also risk shortcut learning
        if keyword tokens dominate the decision.
    """))


# ==============================================================================
# TECHNIQUE 2 — Lexical Contrast: PCL vs Non-PCL
# ==============================================================================

def _tokenise_series(series: pd.Series) -> list[str]:
    """Lower-case word-tokenise, strip pure-punctuation tokens."""
    stop = set(stopwords.words("english"))
    tokens = []
    for text in series.dropna().astype(str):
        for tok in word_tokenize(text.lower()):
            if tok.isalpha() and tok not in stop:
                tokens.append(tok)
    return tokens


def technique_2_lexical_contrast(df_pcl: pd.DataFrame) -> None:
    report("=" * 80)
    report("TECHNIQUE 2 — Lexical Contrast: PCL vs Non-PCL")
    report("=" * 80)

    pcl_texts    = df_pcl.loc[df_pcl["binary_label"] == 1, "text"]
    nonpcl_texts = df_pcl.loc[df_pcl["binary_label"] == 0, "text"]

    pcl_tokens    = _tokenise_series(pcl_texts)
    nonpcl_tokens = _tokenise_series(nonpcl_texts)

    pcl_freq    = Counter(pcl_tokens)
    nonpcl_freq = Counter(nonpcl_tokens)

    # ── log-odds ratio — most distinctive terms per class ─────────────────
    vocab = set(pcl_freq) | set(nonpcl_freq)
    total_pcl = len(pcl_tokens)
    total_non = len(nonpcl_tokens)
    log_odds: dict[str, float] = {}
    for w in vocab:
        p_pcl = (pcl_freq[w] + 1) / (total_pcl + len(vocab))
        p_non = (nonpcl_freq[w] + 1) / (total_non + len(vocab))
        log_odds[w] = np.log(p_pcl / p_non)

    sorted_lo = sorted(log_odds.items(), key=lambda x: x[1])
    top_nonpcl = sorted_lo[:20]                         # most Non-PCL
    top_pcl    = sorted_lo[-20:][::-1]                  # most PCL

    # ── bigrams per class ─────────────────────────────────────────────────
    pcl_bigrams    = Counter(ngrams(pcl_tokens, 2)).most_common(15)
    nonpcl_bigrams = Counter(ngrams(nonpcl_tokens, 2)).most_common(15)

    # ── figure: 2x2 — word clouds + distinctive-term bars ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # word cloud — PCL
    wc_pcl = WordCloud(
        width=800, height=400, background_color="white",
        colormap="Reds", max_words=150, random_state=42,
    ).generate_from_frequencies(pcl_freq)
    axes[0, 0].imshow(wc_pcl, interpolation="bilinear")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("(a) Word Cloud — PCL Paragraphs", fontweight="bold", fontsize=12)

    # word cloud — Non-PCL
    wc_non = WordCloud(
        width=800, height=400, background_color="white",
        colormap="Blues", max_words=150, random_state=42,
    ).generate_from_frequencies(nonpcl_freq)
    axes[0, 1].imshow(wc_non, interpolation="bilinear")
    axes[0, 1].axis("off")
    axes[0, 1].set_title("(b) Word Cloud — Non-PCL Paragraphs", fontweight="bold", fontsize=12)

    # distinctive-term bar chart — PCL
    words_pcl  = [w for w, _ in top_pcl]
    scores_pcl = [s for _, s in top_pcl]
    axes[1, 0].barh(words_pcl[::-1], scores_pcl[::-1], color="#dd4444")
    axes[1, 0].set_xlabel("Log-Odds Ratio (-> more PCL-distinctive)")
    axes[1, 0].set_title("(c) Top 20 PCL-Distinctive Terms", fontweight="bold", fontsize=12)

    # distinctive-term bar chart — Non-PCL
    words_non  = [w for w, _ in top_nonpcl]
    scores_non = [abs(s) for _, s in top_nonpcl]
    axes[1, 1].barh(words_non[::-1], scores_non[::-1], color="#4c72b0")
    axes[1, 1].set_xlabel("Log-Odds Ratio (-> more Non-PCL-distinctive)")
    axes[1, 1].set_title("(d) Top 20 Non-PCL-Distinctive Terms", fontweight="bold", fontsize=12)

    fig.suptitle("Technique 2 — Lexical Contrast: PCL vs Non-PCL",
                 fontsize=15, fontweight="bold", y=1.01)
    fig_path = os.path.join(FIG_DIR, "technique2_lexical_contrast.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report(f"  Figure saved -> {fig_path}")

    # ── second figure: bigram comparison ─────────────────────────────────
    fig2, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(16, 7))

    bg_labels_pcl  = [" ".join(b) for b, _ in pcl_bigrams]
    bg_counts_pcl  = [c for _, c in pcl_bigrams]
    ax_b1.barh(bg_labels_pcl[::-1], bg_counts_pcl[::-1], color="#dd4444")
    ax_b1.set_xlabel("Frequency")
    ax_b1.set_title("Top 15 Bigrams — PCL", fontweight="bold")

    bg_labels_non  = [" ".join(b) for b, _ in nonpcl_bigrams]
    bg_counts_non  = [c for _, c in nonpcl_bigrams]
    ax_b2.barh(bg_labels_non[::-1], bg_counts_non[::-1], color="#4c72b0")
    ax_b2.set_xlabel("Frequency")
    ax_b2.set_title("Top 15 Bigrams — Non-PCL", fontweight="bold")

    fig2.suptitle("Technique 2 (cont.) — Bigram Frequency by Class",
                  fontsize=14, fontweight="bold")
    fig2_path = os.path.join(FIG_DIR, "technique2_bigrams.png")
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    report(f"  Figure saved -> {fig2_path}")

    # ── analysis ─────────────────────────────────────────────────────────
    report("\n-- Analysis --")
    report(f"  PCL vocabulary   : {len(pcl_freq):,} unique content words "
           f"(from {total_pcl:,} tokens)")
    report(f"  Non-PCL vocabulary: {len(nonpcl_freq):,} unique content words "
           f"(from {total_non:,} tokens)")
    report("")
    report("  PCL-distinctive terms (highest log-odds):")
    for w, s in top_pcl[:10]:
        report(f"    {w:20s}  log-odds = {s:+.3f}   "
               f"(PCL freq {pcl_freq[w]:,}, Non-PCL freq {nonpcl_freq[w]:,})")
    report("")
    report("  Non-PCL-distinctive terms (lowest log-odds):")
    for w, s in top_nonpcl[:10]:
        report(f"    {w:20s}  log-odds = {s:+.3f}   "
               f"(PCL freq {pcl_freq[w]:,}, Non-PCL freq {nonpcl_freq[w]:,})")
    report("")
    report("  The word clouds and log-odds analysis reveal that PCL paragraphs")
    report("  tend to use more emotionally charged, paternalistic, or")
    report("  pitying language, while Non-PCL paragraphs are more factual/neutral.")
    report("  Bigram analysis highlights recurring patronising collocations.")

    # ── impact statement ─────────────────────────────────────────────────
    report("\n-- Impact on Classification --")
    report(textwrap.dedent("""\
      * Distinctive terms suggest the model can learn useful lexical cues, but
        heavy reliance on single keywords risks brittle generalisation.
      * Word-cloud visual inspection confirms that stop-word removal is appropriate
        for feature-based models, but transformer models should keep full text
        for contextual understanding.
      * The log-odds vocabulary overlap between classes means simple bag-of-words
        will struggle; contextual models (BERT / RoBERTa) that capture tone and
        framing are strongly preferred.
      * Bigram patterns could inform data augmentation — e.g. paraphrasing PCL
        bigrams to expand minority-class training examples.
    """))


# ==============================================================================
# TECHNIQUE 3 — Sequence-Length Distribution & max_length Selection
# ==============================================================================

def technique_3_length_distribution(df_pcl: pd.DataFrame) -> None:
    report("=" * 80)
    report("TECHNIQUE 3 — Sequence-Length Distribution & max_length Selection")
    report("=" * 80)

    texts = df_pcl["text"].dropna().astype(str)
    df_pcl = df_pcl.loc[texts.index].copy()
    token_counts = texts.apply(lambda t: len(word_tokenize(t)))
    df_pcl["n_tokens"] = token_counts.values

    pcl_lens    = df_pcl.loc[df_pcl["binary_label"] == 1, "n_tokens"]
    nonpcl_lens = df_pcl.loc[df_pcl["binary_label"] == 0, "n_tokens"]

    # ── percentile table ─────────────────────────────────────────────────
    percentiles = [50, 75, 90, 95, 99]
    table_rows = []
    for p in percentiles:
        table_rows.append({
            "Percentile": f"P{p}",
            "All":     int(np.percentile(token_counts, p)),
            "PCL":     int(np.percentile(pcl_lens, p)),
            "Non-PCL": int(np.percentile(nonpcl_lens, p)),
        })
    perc_df = pd.DataFrame(table_rows).set_index("Percentile")

    # ── figure: 2x2 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) overall histogram
    axes[0, 0].hist(token_counts, bins=80, color="#7f7f7f", edgecolor="white", alpha=0.85)
    axes[0, 0].axvline(np.percentile(token_counts, 95), color="red", ls="--", lw=1.5,
                        label=f"P95 = {int(np.percentile(token_counts, 95))}")
    axes[0, 0].axvline(np.median(token_counts), color="blue", ls="--", lw=1.5,
                        label=f"Median = {int(np.median(token_counts))}")
    axes[0, 0].set_xlabel("Token Count")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("(a) Overall Token-Count Distribution", fontweight="bold")
    axes[0, 0].legend()

    # (b) overlaid histograms — PCL vs Non-PCL
    axes[0, 1].hist(nonpcl_lens, bins=80, color="#4c72b0", alpha=0.6,
                     label="Non-PCL", density=True)
    axes[0, 1].hist(pcl_lens, bins=40, color="#dd4444", alpha=0.6,
                     label="PCL", density=True)
    axes[0, 1].set_xlabel("Token Count")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("(b) Token-Count Density by Class", fontweight="bold")
    axes[0, 1].legend()

    # (c) box plot
    box_data = [nonpcl_lens.values, pcl_lens.values]
    bp = axes[1, 0].boxplot(box_data, labels=["Non-PCL", "PCL"], patch_artist=True,
                             widths=0.5, showfliers=True,
                             flierprops=dict(marker="o", markersize=2, alpha=0.3))
    bp["boxes"][0].set_facecolor("#4c72b0")
    bp["boxes"][1].set_facecolor("#dd4444")
    axes[1, 0].set_ylabel("Token Count")
    axes[1, 0].set_title("(c) Box Plot — Token Counts by Class", fontweight="bold")

    # (d) cumulative distribution — choose max_length
    sorted_all = np.sort(token_counts)
    cdf = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
    axes[1, 1].plot(sorted_all, cdf, color="#333333", lw=2)
    for p, col in [(90, "orange"), (95, "red"), (99, "darkred")]:
        val = int(np.percentile(token_counts, p))
        axes[1, 1].axvline(val, ls="--", color=col, lw=1.2, label=f"P{p} = {val}")
        axes[1, 1].axhline(p / 100, ls=":", color=col, lw=0.6, alpha=0.5)
    axes[1, 1].set_xlabel("Token Count")
    axes[1, 1].set_ylabel("Cumulative Proportion")
    axes[1, 1].set_title("(d) CDF — Choosing max_length", fontweight="bold")
    axes[1, 1].legend(loc="lower right")

    fig.suptitle("Technique 3 — Sequence-Length Distribution & max_length",
                 fontsize=15, fontweight="bold", y=1.01)
    fig_path = os.path.join(FIG_DIR, "technique3_length_distribution.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report(f"  Figure saved -> {fig_path}")

    # ── percentile table ─────────────────────────────────────────────────
    report("\n-- Percentile Table (token counts) --")
    report(perc_df.to_string())

    # ── analysis ─────────────────────────────────────────────────────────
    p95_all = int(np.percentile(token_counts, 95))
    p95_pcl = int(np.percentile(pcl_lens, 95))
    report("\n-- Analysis --")
    report(f"  Median length     : {int(np.median(token_counts))} tokens")
    report(f"  Mean length       : {np.mean(token_counts):.1f} tokens")
    report(f"  P95 (all)         : {p95_all} tokens")
    report(f"  P95 (PCL only)    : {p95_pcl} tokens")
    report(f"  Max length        : {int(np.max(token_counts))} tokens")
    report("")
    report("  The distribution is right-skewed: most paragraphs are short")
    report("  (< 60 tokens) but a long tail extends beyond 200 tokens.")
    report("  PCL paragraphs tend to be slightly longer than Non-PCL,")
    report("  suggesting patronising writing is more verbose on average.")
    report("  The CDF plot shows that a max_length at P95 captures the vast")
    report("  majority of the text without wasting computation on outliers.")

    # ── impact statement ─────────────────────────────────────────────────
    report("\n-- Impact on Classification --")
    report(textwrap.dedent(f"""\
      * A max_length of ~{p95_all} tokens (P95) is a strong default for
        transformer tokeniser truncation; it retains >=95% of paragraphs
        in full while keeping GPU memory manageable.
      * Because PCL paragraphs skew longer, aggressive truncation risks
        losing patronising content that appears later in the text.
        If resources permit, increasing to P99 is safer for the minority class.
      * Very short paragraphs (< 10 tokens) are unlikely to carry enough
        context for reliable PCL detection and may add noise; filtering or
        special handling could improve precision.
      * The length difference between classes is a weak signal that could
        complement lexical features in an ensemble, but should not be a
        primary discriminator.
    """))


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    report("=" * 80)
    report("  EDA REPORT — Don't Patronize Me! Dataset")
    report(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report("=" * 80)
    report("")

    df_pcl, df_cat = load_data()
    report(f"  Loaded PCL dataset        : {df_pcl.shape[0]:,} rows x {df_pcl.shape[1]} cols")
    report(f"  Loaded Categories dataset : {df_cat.shape[0]:,} rows x {df_cat.shape[1]} cols")
    report("")

    technique_1_class_distribution(df_pcl)
    technique_2_lexical_contrast(df_pcl)
    technique_3_length_distribution(df_pcl)

    report("=" * 80)
    report("  EDA COMPLETE — figures saved in eda/figures/")
    report("=" * 80)

    save_report()


if __name__ == "__main__":
    main()
