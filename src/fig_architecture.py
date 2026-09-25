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


def box(ax, xy, w, h, text, edgecolor="black", fontsize=7.5, fontweight="normal"):
    ax.add_patch(plt.Rectangle(xy, w, h, fill=False, edgecolor=edgecolor, linewidth=1.1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=edgecolor)


def arrow(ax, xy_from, xy_to, color="black"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="->", lw=1.1, color=color))


def main():
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("CNN-and-hybrid-classifier pipeline and independently evaluated architectures", fontsize=10)

    # --- shared input box ---
    box(ax, (2.2, 9.0), 5.6, 0.7,
        "Raw EEG epoch (19 channels x 3840 samples, 128 Hz)\nper-epoch normalization (Sec. 3.4-3.5)",
        fontsize=7.5)
    arrow(ax, (3.8, 9.0), (2.6, 8.35))
    arrow(ax, (6.2, 9.0), (8.0, 8.35))

    # ================= LEFT COLUMN: CNN + hybrid classifiers =================
    box(ax, (0.3, 7.6), 4.6, 0.7,
        "CNN (Sec. 3.6): 2 spatial + 2 temporal conv blocks\n-> 64-unit dense -> 32-unit dense", fontsize=7.5)
    arrow(ax, (1.3, 7.6), (1.3, 6.95))
    arrow(ax, (3.9, 7.6), (3.9, 6.95))

    box(ax, (0.3, 6.25), 2.2, 0.7, "Sigmoid output\n(plain CNN prediction)", fontsize=7)
    box(ax, (2.7, 6.25), 2.2, 0.7, "64-unit dense-layer\nfeature vector", fontsize=7)

    arrow(ax, (3.8, 6.25), (2.3, 5.6))
    arrow(ax, (3.8, 6.25), (3.8, 5.6))

    box(ax, (0.3, 4.9), 2.6, 0.7,
        "CORAL alignment (Sec. 3.18)\ntransductive: uses outer-test\nfeature covariance, no labels",
        fontsize=6.7)
    box(ax, (3.1, 4.9), 2.0, 0.7, "Unaligned\nfeatures", fontsize=7)

    arrow(ax, (1.6, 4.9), (1.6, 4.25))
    arrow(ax, (4.1, 4.9), (4.1, 4.25))

    box(ax, (0.3, 3.55), 2.6, 0.7,
        "Logistic regression on\nCORAL-aligned features\n(CNN+LR_withCORAL)", fontsize=6.7)
    box(ax, (3.1, 3.55), 2.0, 0.7,
        "LR, RBF-SVM, linear SVM,\nrandom forest, k-NN,\nGaussian NB (Sec. 3.8)", fontsize=6.3)

    arrow(ax, (4.1, 3.55), (4.1, 2.9))
    box(ax, (3.1, 2.2), 2.0, 0.7, "Hybrid-classifier\npredictions", fontsize=7)

    # ================= RIGHT COLUMN: independent architectures =================
    right_x = 6.6
    right_w = 3.1
    labels = [
        ("EEGNet (Sec. 3.9)", "depthwise + separable\nconvolutions"),
        ("ShallowConvNet (Sec. 3.10)", "temporal + spatial conv,\nsquare / log nonlinearity"),
        ("DeepConvNet (Sec. 3.11)", "4 conv-pool blocks,\n25-50-100-200 filters"),
    ]
    y = 7.6
    for title, detail in labels:
        box(ax, (right_x, y), right_w, 0.7, f"{title}\n{detail}", fontsize=6.8)
        arrow(ax, (right_x + right_w / 2, y), (right_x + right_w / 2, y - 0.65))
        box(ax, (right_x + 0.55, y - 1.35), right_w - 1.1, 0.7, "Prediction", fontsize=7)
        y -= 2.15

    ax.text(0.05, 0.35,
            "Solid boxes: components described in Section 3. Left column shares one CNN feature extractor\n"
            "across all classical classifiers (Sec. 3.7-3.8, 3.18); right column architectures are each trained\n"
            "and evaluated independently, under the identical nested cross-validation design (Fig. 1).",
            fontsize=6.5, va="bottom")

    fig.savefig("manuscript/FIGURES/fig_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote manuscript/FIGURES/fig_architecture.png")


if __name__ == "__main__":
    main()
