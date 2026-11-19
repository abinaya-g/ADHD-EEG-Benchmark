"""Regenerates manuscript Figure 4 (subject-level confusion matrices) at
publication quality (2x5 grid, 300 dpi, large fonts) directly from
results_final/predictions/all_predictions.csv. Same aggregation as Table 3:
per-subject mean probability over all epochs and repetitions, threshold 0.5."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["CNN", "CNN+LR", "CNN+NLSVM", "CNN+RF", "CNN+GNB", "CNN+KNN",
         "CNN+LinearSVM", "EEGNet", "ShallowConvNet", "DeepConvNet"]


def main(pred="results_final/predictions/all_predictions.csv",
         out="manuscript/FIGURES/fig6_confusion.png"):
    d = pd.read_csv(pred)
    d = d[d.preprocessing == "128Hz"]
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 4, figsize=(11, 9.4))
    for ax in axes.ravel()[len(ORDER):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), ORDER):
        s = d[d.model == m].groupby("subject_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))
        yh = (s.p >= 0.5).astype(int)
        cm = np.array([[((s.y == i) & (yh == j)).sum() for j in (0, 1)] for i in (0, 1)])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}\n({100 * frac:.0f}%)", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white" if frac > 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "ADHD"], fontsize=13)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "ADHD"], fontsize=13, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=14); ax.set_ylabel("True", fontsize=14)
        ba = 0.5 * (cm[0, 0] / cm[0].sum() + cm[1, 1] / cm[1].sum())
        ax.set_title(f"{m}\nBA = {ba:.3f}", fontsize=14, fontweight="bold")
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
