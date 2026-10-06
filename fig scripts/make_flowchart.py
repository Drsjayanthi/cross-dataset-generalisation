import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "text.color": "black",
    "axes.labelcolor": "black"
})

INK = "black"
NEUTRAL_FILL = "#F3F4F0"
NEUTRAL_EDGE = "#4A4A4A"
A_FILL = "#DCEAE6"   # Study A - teal tint
A_EDGE = "#1F6E64"
B_FILL = "#EFE1DA"   # Study B - rust tint
B_EDGE = "#8E5C4E"
ARROW_COLOR = "#3A3A3A"

fig, ax = plt.subplots(figsize=(7.0, 7.9), dpi=600)
ax.set_xlim(0, 10)
ax.set_ylim(2.25, 13.3)
ax.axis("off")


def box(x0, x1, y0, y1, lines, fill=NEUTRAL_FILL, edge=NEUTRAL_EDGE, fontsize=7.6,
        header=None, header_size=8.0, align="center", lw=0.9):
    b = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                        boxstyle="round,pad=0.0,rounding_size=0.09",
                        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(b)
    cx = (x0 + x1) / 2
    top = y1 - 0.16
    if header:
        ax.text(cx, top, header, ha="center", va="top", fontsize=header_size,
                 weight="bold", color=INK, zorder=3)
        top -= 0.30
    ha = "center" if align == "center" else "left"
    tx = cx if align == "center" else x0 + 0.16
    for line in lines:
        ax.text(tx, top, line, ha=ha, va="top", fontsize=fontsize, color=INK, zorder=3)
        top -= 0.235
    return b


def arrow(xy_from, xy_to, lw=1.1):
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
                         linewidth=lw, color=ARROW_COLOR, zorder=1,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def wrap(text, width):
    return textwrap.wrap(text, width=width)


# ---------------- Stage 1: candidate datasets ----------------
box(0.8, 9.2, 12.55, 13.15,
    ["RESISC45  \u00b7  UCMerced Land Use  \u00b7  AID  \u00b7  EuroSAT-RGB   (n = 70,600 images)"],
    header="Four candidate benchmark datasets", fontsize=7.6)
arrow((5, 12.55), (5, 12.05))

# ---------------- Stage 2: QC ----------------
box(0.8, 9.2, 11.30, 12.05,
    ["Corrupt-file check  \u00b7  MD5 exact-duplicate detection  \u00b7  pHash near-duplicate screening"],
    header="Quality control  (\u00a7 2.2)", fontsize=7.6)
arrow((5, 11.30), (5, 10.80))

# ---------------- Stage 3: comparability assessment ----------------
box(1.3, 8.7, 10.05, 10.80,
    ["Ground sampling distance  \u00b7  label taxonomy overlap"],
    header="Dataset comparability assessment  (\u00a7 2.2)", fontsize=7.6)

# branch arrows
arrow((3.2, 10.05), (2.65, 9.55))
arrow((6.8, 10.05), (7.35, 9.55))

# ---------------- Stage 4: branch outcome ----------------
box(0.8, 4.5, 8.75, 9.55,
    ["Judged directly comparable"],
    fill=A_FILL, edge=A_EDGE, header="RESISC45 + UCMerced + AID", fontsize=7.6)
box(5.5, 9.2, 8.75, 9.55,
    ["Judged not comparable (GSD, taxonomy)"],
    fill=B_FILL, edge=B_EDGE, header="EuroSAT-RGB", fontsize=7.6)

arrow((2.65, 8.75), (2.65, 8.25))
arrow((7.35, 8.75), (7.35, 8.25))

# ---------------- Stage 5: label space ----------------
box(0.8, 4.5, 7.45, 8.25,
    ["12-class harmonised label space", "n = 14,500 images  (\u00a7 2.3)"],
    fill=A_FILL, edge=A_EDGE, fontsize=7.6)
box(5.5, 9.2, 7.45, 8.25,
    ["Native 10-class taxonomy", "n = 27,000 images"],
    fill=B_FILL, edge=B_EDGE, fontsize=7.6)

arrow((2.65, 7.45), (2.65, 7.00))
arrow((7.35, 7.45), (7.35, 7.00))

# ---------------- Stage 6: evaluation designs ----------------
box(0.8, 4.5, 5.40, 7.00,
    ["A1   Leave-one-dataset-out (LODO)",
     "A2   Pooled 5-fold CV",
     "A3   Native-taxonomy 5-fold CV",
     "\u00d7 5 architectures each  (\u00a7 2.4\u20132.5)",
     "Preprocessing: resize + normalise  (\u00a7 2.5)"],
    fill=A_FILL, edge=A_EDGE, header="Study A \u2014 cross-dataset design", fontsize=7.4, align="left")
box(5.5, 9.2, 5.40, 7.00,
    ["B1   5-fold CV",
     "B2   Held-out test (20%)",
     "\u00d7 5 architectures each  (\u00a7 2.4\u20132.5)",
     "Preprocessing: resize + normalise  (\u00a7 2.5)"],
    fill=B_FILL, edge=B_EDGE, header="Study B \u2014 standalone design", fontsize=7.4, align="left")

arrow((2.65, 5.40), (4.85, 4.90))
arrow((7.35, 5.40), (5.15, 4.90))

# ---------------- Stage 7: statistical analysis ----------------
box(0.8, 9.2, 4.00, 4.90,
    ["Bootstrap 95% CI (2,000 resamples)  \u00b7  McNemar's test  \u00b7  TOST equivalence, Holm-corrected"],
    header="Statistical analysis  (\u00a7 2.6)", fontsize=7.6)
arrow((5, 4.00), (5, 3.40))

# ---------------- Stage 8: results ----------------
box(2.3, 7.7, 2.60, 3.40,
    ["Results, discussion, and conclusion  (\u00a7 3\u20135)"],
    fontsize=7.6)

plt.tight_layout(pad=0.3)
plt.savefig(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\fig1_pipeline.png",
    dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close()
print("Flowchart saved.")
