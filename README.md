# PCL Detection

Binary classification of Patronising and Condescending Language.
SemEval 2022 Task 4, Subtask 1.

## Approach

Weighted soft-vote ensemble of three fine-tuned transformers:

- RoBERTa-base
- ModernBERT-base
- MPNet-base

Class-weighted cross-entropy loss handles the ~9.5:1 class imbalance.
Text is cleaned of HTML tags and entities before tokenisation.

## Repository Structure

```
.
├── BestModel/
│   └── train.py                  # model training, evaluation, prediction
├── Dont_Patronize_Me_Trainingset/
│   ├── dontpatronizeme_pcl.tsv   # main dataset
│   ├── dontpatronizeme_categories.tsv
│   ├── dev_semeval_parids-labels.csv
│   └── task4_test.tsv            # official test set (no labels)
├── dev.txt                       # official dev set predictions
├── test.txt                      # official test set predictions
├── eda_analysis.py               # exploratory data analysis script
├── eda_results.txt               # EDA output
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for training. CPU works for inference but will be slow.

## Usage

Train all models from scratch:

```bash
cd BestModel
python train.py
```

Reload from saved checkpoints and evaluate:

```bash
cd BestModel
python train.py --reload
```

Both modes produce `dev.txt` and `test.txt` at the repository root
(one prediction per line: 0 = Non-PCL, 1 = PCL).

## Evaluation

Primary metric: F1 score on the positive class (PCL).
Baseline: 0.48 F1 (dev), 0.49 F1 (test).

## EDA

Run from the repository root:

```bash
python eda_analysis.py
```

Results are written to `eda_results.txt`.
