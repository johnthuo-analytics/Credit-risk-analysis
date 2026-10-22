"""Metrics and diagnostic plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary_metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"roc_auc": auc, "gini": 2 * auc - 1,
            "ks": ks_statistic(y, p), "pr_auc": average_precision_score(y, p)}


def best_threshold(y, p, c_fn=COST_FALSE_NEGATIVE, c_fp=COST_FALSE_POSITIVE):
    """Threshold minimising expected cost; defaulting at 0.5 ignores asymmetric costs."""
    grid = np.linspace(0.02, 0.95, 94)
    costs = []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
        costs.append(fn * c_fn + fp * c_fp)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def threshold_report(y, p, t) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
    return {"threshold": t, "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
            "approval_rate": (tn + fn) / len(y), "bad_rate_in_approved": fn / max(tn + fn, 1)}


def plot_diagnostics(y, p, name, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, lw=2, label=f"PR-AUC = {average_precision_score(y, p):.3f}")
    ax[1].axhline(np.mean(y), color="k", ls="--", lw=1, label="Baseline")
    ax[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    ax[1].legend()
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax[2].plot(mean_p, frac, "o-", lw=2)
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[2].set(title="Calibration", xlabel="Predicted probability", ylabel="Observed default rate")
    fig.suptitle(f"{name} - hold-out test set", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
