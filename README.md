# Edge Anomaly Detection

Exploring **robust and efficient visual anomaly detection for edge devices** — how well established anomaly detection methods hold up on visually difficult materials, and whether they can run on resource-constrained hardware such as the Raspberry Pi 5.

| Project | Topic | Status |
|---|---|---|
| [1](#project-1--baseline-reproduction) | Baseline reproduction: PatchCore vs PaDiM on MVTec AD | ✅ Done |
| 2 | Failure analysis on reflective / transparent materials | 🔄 Next |
| 3 | Quantization and deployment on Raspberry Pi 5 | ⏳ Planned |
| 4 | Logical anomalies on MVTec LOCO (stretch) | ⏳ Planned |

---

## Project 1 — Baseline Reproduction

### Goal
Reproduce two established feature-embedding anomaly detection methods, **PatchCore** and **PaDiM**, on the MVTec AD benchmark, and understand *why* they differ rather than only *how much*.

### Setup
- **Library:** [anomalib](https://github.com/open-edge-platform/anomalib) 2.x, default settings for both models
- **Dataset:** MVTec AD, categories `bottle` and `screw`
- **Hardware:** Kaggle notebook, single GPU
- **Backbones (anomalib defaults):** PatchCore → WideResNet50, PaDiM → ResNet18

Both methods are trained only on defect-free images. At test time, each image patch is compared against what "normal" looks like; patches that deviate strongly are flagged as anomalous, producing an anomaly heatmap.

### Results

| Method | Category | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|---|---|---|---|---|---|
| PatchCore | bottle | 1.000 | 0.992 | 0.986 | 0.727 |
| PaDiM | bottle | 1.000 | 0.992 | 0.980 | 0.691 |
| PatchCore | screw | **0.962** | **0.950** | **0.988** | **0.372** |
| PaDiM | screw | 0.818 | 0.874 | 0.976 | 0.221 |

<!-- Add heatmap examples, e.g.:
![PatchCore vs PaDiM on screw](project1/heatmaps/screw_comparison.png)
-->

### Findings

**1. `bottle` is saturated.** Both methods reach an image AUROC of 1.0, so this category cannot distinguish between them.

**2. PaDiM breaks down when object pose varies.** On `screw`, where screws are photographed at random rotations, PatchCore stays high (0.962) while PaDiM drops to 0.818. This is consistent with PaDiM's core assumption: it models the normal feature distribution *separately for each grid position*, which only works if the same part of the object always appears at the same location. PatchCore instead compares each test patch against *all* normal patches regardless of position, making it more tolerant to misalignment. I predicted this gap before running the `screw` experiment; the result matched the prediction.

**3. Pixel AUROC can hide poor localization.** On `screw`, both methods keep a pixel AUROC above 0.97, yet their pixel F1 is low (0.372 and 0.221). Screw defects are tiny, so the vast majority of pixels are easy-to-classify background, which inflates AUROC. Region-based metrics such as PRO/AUPRO are better suited to small defects.

### Limitations
- **Backbone confound.** PatchCore uses a larger backbone (WideResNet50) than PaDiM (ResNet18), so part of the gap may come from feature quality rather than the method itself. A fair comparison would match backbones.
- Only two categories were evaluated, with default hyperparameters and a single run each.

### Open question → Project 3
PatchCore's advantage comes with a cost: a larger backbone and a memory bank of normal features that must be stored and searched at inference time. Whether this advantage survives on an edge device, under quantization and tight memory and latency budgets, is the question Project 3 will measure.

### Reproduce
1. Create a Kaggle notebook with GPU enabled and internet on.
2. Add the MVTec AD dataset as an input.
3. Run `project1/notebook.ipynb`.

---

## Repository Structure

```
.
├── README.md
└── project1/
    ├── notebook.ipynb
    ├── results.csv
    └── heatmaps/
```
