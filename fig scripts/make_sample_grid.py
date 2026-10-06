import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
})

# ---------------- systematically selected sample files (first file per class/dataset, sorted) ----------------
# Order matches Table 3 (12-class harmonised label space).
ROOT_RESISC = r"D:\SJ\UVMDATA\NWPU-RESISC45"
ROOT_UCM = r"D:\SJ\UVMDATA\UCMerced_LandUse\UCMerced_landuse"
ROOT_AID = r"D:\SJ\UVMDATA\AID\AID"

PRIMARY_CLASSES = [
    ("agricultural\nfarmland", {
        "RESISC45": ROOT_RESISC + r"\circular_farmland\circular_farmland_001.jpg",
        "UCMerced": ROOT_UCM + r"\agricultural\agricultural00.tif",
        "AID": ROOT_AID + r"\Farmland\farmland_1.jpg",
    }),
    ("baseball\nfield", {
        "RESISC45": ROOT_RESISC + r"\baseball_diamond\baseball_diamond_001.jpg",
        "UCMerced": ROOT_UCM + r"\baseballdiamond\baseballdiamond00.tif",
        "AID": ROOT_AID + r"\BaseballField\baseballfield_1.jpg",
    }),
    ("beach", {
        "RESISC45": ROOT_RESISC + r"\beach\beach_001.jpg",
        "UCMerced": ROOT_UCM + r"\beach\beach00.tif",
        "AID": ROOT_AID + r"\Beach\beach_1.jpg",
    }),
    ("dense\nresidential", {
        "RESISC45": ROOT_RESISC + r"\dense_residential\dense_residential_001.jpg",
        "UCMerced": ROOT_UCM + r"\denseresidential\denseresidential00.tif",
        "AID": ROOT_AID + r"\DenseResidential\denseresidential_1.jpg",
    }),
    ("forest", {
        "RESISC45": ROOT_RESISC + r"\forest\forest_001.jpg",
        "UCMerced": ROOT_UCM + r"\forest\forest00.tif",
        "AID": ROOT_AID + r"\Forest\forest_1.jpg",
    }),
    ("harbor\nport", {
        "RESISC45": ROOT_RESISC + r"\harbor\harbor_001.jpg",
        "UCMerced": ROOT_UCM + r"\harbor\harbor00.tif",
        "AID": ROOT_AID + r"\Port\port_1.jpg",
    }),
    ("medium\nresidential", {
        "RESISC45": ROOT_RESISC + r"\medium_residential\medium_residential_001.jpg",
        "UCMerced": ROOT_UCM + r"\mediumresidential\mediumresidential00.tif",
        "AID": ROOT_AID + r"\MediumResidential\mediumresidential_1.jpg",
    }),
    ("overpass\nviaduct", {
        "RESISC45": ROOT_RESISC + r"\overpass\overpass_001.jpg",
        "UCMerced": ROOT_UCM + r"\overpass\overpass00.tif",
        "AID": ROOT_AID + r"\Viaduct\viaduct_1.jpg",
    }),
    ("parking\nlot", {
        "RESISC45": ROOT_RESISC + r"\parking_lot\parking_lot_001.jpg",
        "UCMerced": ROOT_UCM + r"\parkinglot\parkinglot00.tif",
        "AID": ROOT_AID + r"\Parking\parking_1.jpg",
    }),
    ("river", {
        "RESISC45": ROOT_RESISC + r"\river\river_001.jpg",
        "UCMerced": ROOT_UCM + r"\river\river00.tif",
        "AID": ROOT_AID + r"\River\river_1.jpg",
    }),
    ("sparse\nresidential", {
        "RESISC45": ROOT_RESISC + r"\sparse_residential\sparse_residential_001.jpg",
        "UCMerced": ROOT_UCM + r"\sparseresidential\sparseresidential00.tif",
        "AID": ROOT_AID + r"\SparseResidential\sparseresidential_1.jpg",
    }),
    ("storage\ntanks", {
        "RESISC45": ROOT_RESISC + r"\storage_tank\storage_tank_001.jpg",
        "UCMerced": ROOT_UCM + r"\storagetanks\storagetanks00.tif",
        "AID": ROOT_AID + r"\StorageTanks\storagetanks_1.jpg",
    }),
]
PRIMARY_ROWS = ["RESISC45", "UCMerced", "AID"]
N_CLASSES = len(PRIMARY_CLASSES)

INK = "black"

# ---------------- figure: landscape orientation, class names above images (single panel) ----------------
fig = plt.figure(figsize=(9.5, 3.05), dpi=600)

col_widths = [0.62] + [1] * N_CLASSES  # label column + 12 class columns
gs = gridspec.GridSpec(
    nrows=4, ncols=1 + N_CLASSES, figure=fig,
    height_ratios=[0.42, 1, 1, 1],
    width_ratios=col_widths,
    hspace=0.16, wspace=0.08,
)

for j, (cls_label, _) in enumerate(PRIMARY_CLASSES):
    ax = fig.add_subplot(gs[0, j + 1])
    ax.axis("off")
    ax.text(0.5, 0.0, cls_label, ha="center", va="bottom", fontsize=6.8, weight="bold",
            color=INK, linespacing=1.15)

for i, row_name in enumerate(PRIMARY_ROWS):
    ax_lab = fig.add_subplot(gs[i + 1, 0])
    ax_lab.axis("off")
    ax_lab.text(0.95, 0.5, row_name, ha="right", va="center", fontsize=8.2, color=INK)
    for j, (cls_label, files) in enumerate(PRIMARY_CLASSES):
        ax = fig.add_subplot(gs[i + 1, j + 1])
        im = Image.open(files[row_name]).convert("RGB")
        ax.imshow(im, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#999999")
            spine.set_linewidth(0.4)

plt.savefig(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\fig3_samples_12class.png",
    dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close()
print("Sample-image grid saved (landscape, all 12 harmonised classes, single panel).")
