"""Synthetic, group-aware machine learning for biological phenotyping."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 47
OUTPUT_DIR = Path("outputs")
FEATURES = [f"reference_measurement_{index}" for index in range(1, 7)]


def make_synthetic_data(n_batches: int = 24, observations_per_batch: int = 18) -> pd.DataFrame:
    """Create a deterministic dataset with batch-level and observation-level variation."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for batch in range(n_batches):
        batch_shift = rng.normal(0, 0.45, size=len(FEATURES))
        for observation in range(observations_per_batch):
            measurements = rng.normal(0, 1, size=len(FEATURES)) + batch_shift
            latent_score = (
                1.25 * measurements[0]
                - 0.9 * measurements[1]
                + 0.7 * measurements[2]
                + 0.45 * measurements[3] * measurements[4]
                + rng.normal(0, 0.85)
            )
            outcome = int(latent_score > 0)
            rows.append(
                {
                    "sample_id": f"B{batch + 1:02d}_S{observation + 1:02d}",
                    "batch_id": f"batch_{batch + 1:02d}",
                    **dict(zip(FEATURES, measurements, strict=True)),
                    "phenotype_outcome": outcome,
                }
            )

    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Summarise observed and predicted outcome frequency by probability bin."""
    frame = pd.DataFrame({"outcome": y_true, "probability": probability})
    frame["probability_bin"] = pd.cut(
        frame["probability"], bins=np.linspace(0, 1, 6), include_lowest=True
    )
    return (
        frame.groupby("probability_bin", observed=True)
        .agg(
            n=("outcome", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_outcome_rate=("outcome", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = make_synthetic_data()
    data.to_csv(OUTPUT_DIR / "synthetic_phenotypes.csv", index=False)

    x = data[FEATURES]
    y = data["phenotype_outcome"].to_numpy()
    groups = data["batch_id"]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED),
    )
    splitter = GroupKFold(n_splits=6)
    probability = cross_val_predict(
        model,
        x,
        y,
        groups=groups,
        cv=splitter,
        method="predict_proba",
    )[:, 1]

    metrics = {
        "n_observations": int(len(data)),
        "n_batches": int(data["batch_id"].nunique()),
        "roc_auc": round(float(roc_auc_score(y, probability)), 3),
        "brier_score": round(float(brier_score_loss(y, probability)), 3),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    calibration = calibration_table(y, probability)
    calibration.to_csv(OUTPUT_DIR / "calibration.csv", index=False)

    model.fit(x, y)
    coefficients = pd.DataFrame(
        {
            "feature": FEATURES,
            "standardised_coefficient": model.named_steps["logisticregression"].coef_[0],
        }
    ).sort_values("standardised_coefficient", key=np.abs, ascending=False)
    coefficients.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
