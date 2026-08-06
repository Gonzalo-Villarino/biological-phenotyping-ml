# Machine learning for biological phenotyping

This repository is an independent, reproducible demonstration of a grouped
classification workflow for biological phenotyping. It connects synthetic
reference measurements with a binary phenotype outcome while keeping
experimental batches together during validation.

## Why this example exists

Biological datasets often contain repeated measurements, batches, accessions,
or experimental sessions. Randomly splitting individual rows can leak related
observations across training and test sets and make performance look better
than it really is. This example uses group-aware cross-validation to evaluate
generalisation to unseen experimental batches.

## Workflow

1. Generate a deterministic synthetic dataset with six reference measurements,
   a batch identifier, and a phenotype outcome.
2. Build a standardised logistic-regression classifier.
3. Produce out-of-fold predictions with `GroupKFold`.
4. Report ROC-AUC and Brier score.
5. Summarise calibration and model coefficients for interpretation.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

The script writes the synthetic dataset and evaluation outputs to `outputs/`.

## Confidentiality boundary

This is a clean, independently created demonstration. The dataset is generated
entirely in code and does not reproduce or derive from any employer dataset,
source code, protocol, experimental design, unpublished result, client
information, or commercial detail.

## Author

Gonzalo Villarino, PhD  
https://gonzalovillarino.com/
