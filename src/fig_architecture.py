"""Figure 2: CNN-and-hybrid-classifier pipeline, alongside the three
independently evaluated end-to-end architectures (EEGNet, ShallowConvNet,
DeepConvNet). A conceptual/schematic diagram of the architecture already
described in Sections 3.6-3.11 and 3.18 -- not synthesized from result
tables, same category as fig3_nested_cv_schematic.py (see the note at the
top of src/visualization.py). Built procedurally with matplotlib so it
matches that figure's plain box-and-arrow style exactly, rather than a
free-form AI-generated image, since every box and arrow here must match
the methods text precisely.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def box(ax, xy, w, h, text, edgecolor="black", fontsize=11.5, fontweight="normal"):
    ax.add_patch(plt.Rectangle(xy, w, h, fill=False, edgecolor=edgecolor, linewidth=1.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=edgecolor)


def arrow(ax, xy_from, xy_to, color="black"):
    ax.annotate("", xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle="->", lw=1.8, color=color))


def main():
    # drawn 12 x 13.5 in and placed at ~6.3 in width: 11.5 pt text prints at ~6 pt
    fig, ax = plt.subplots(figsize=(12, 13.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")

    box(ax, (2.0, 12.6), 8.0, 1.0,
        "Raw EEG epoch (19 channels x 3840 samples, 128 Hz)\nper-epoch normalization (Sec. 3.4-3.5)")
    arrow(ax, (4.0, 12.6), (3.2, 11.8))
    arrow(ax, (8.0, 12.6), (9.6, 11.8))

    # LEFT: CNN + hybrid classifiers
    box(ax, (0.2, 10.6), 6.2, 1.2, "CNN (Sec. 3.6)\n2 spatial + 2 temporal conv blocks\n-> 64-unit dense -> 32-unit dense")
    arrow(ax, (1.7, 10.6), (1.7, 9.7))
    arrow(ax, (4.9, 10.6), (4.9, 9.7))
    box(ax, (0.2, 8.5), 2.9, 1.2, "Sigmoid output\n(plain CNN\nprediction)")
    box(ax, (3.3, 8.5), 3.1, 1.2, "64-unit dense-\nlayer feature\nvector")
    arrow(ax, (4.85, 8.5), (2.0, 7.5))
    arrow(ax, (4.85, 8.5), (4.85, 7.5))
    box(ax, (0.2, 6.0), 3.5, 1.5, "CORAL alignment\n(Sec. 3.18), transductive:\nouter-test feature\ncovariance, no labels")
    box(ax, (3.9, 6.0), 2.5, 1.5, "Unaligned\nfeatures")
    arrow(ax, (1.95, 6.0), (1.95, 5.0))
    arrow(ax, (5.15, 6.0), (5.15, 5.0))
    box(ax, (0.2, 3.5), 3.5, 1.5, "Logistic regression\non CORAL-aligned\nfeatures\n(CNN+LR_withCORAL)")
    box(ax, (3.9, 3.5), 2.5, 1.5, "LR, RBF-SVM,\nlinear SVM, RF,\nk-NN, GNB\n(Sec. 3.8)")
    arrow(ax, (5.15, 3.5), (5.15, 2.6))
    box(ax, (3.9, 1.5), 2.5, 1.1, "Hybrid-classifier\npredictions")

    # RIGHT: independent architectures
    rx, rw, y = 7.2, 4.6, 10.6
    for title, detail in [("EEGNet (Sec. 3.9)", "depthwise + separable\nconvolutions"),
                          ("ShallowConvNet (Sec. 3.10)", "temporal + spatial conv,\nsquare / log nonlinearity"),
                          ("DeepConvNet (Sec. 3.11)", "4 conv-pool blocks,\n25-50-100-200 filters")]:
        box(ax, (rx, y), rw, 1.3, f"{title}\n{detail}")
        arrow(ax, (rx + rw / 2, y), (rx + rw / 2, y - 0.8))
        box(ax, (rx + 1.0, y - 1.7), rw - 2.0, 0.9, "Prediction")
        y -= 3.1

    ax.text(0.2, 0.9,
            "Left: one CNN feature extractor shared by all classical classifiers (Sec. 3.7-3.8, 3.18).\n"
            "Right: architectures trained and evaluated independently under the identical\n"
            "nested cross-validation design (Fig. 2).", fontsize=11.5, va="top")

    fig.savefig("manuscript/FIGURES/fig_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote manuscript/FIGURES/fig_architecture.png")


if __name__ == "__main__":
    main()
