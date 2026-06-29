# Enhancing Tomato Leaf Disease Detection Using Federated Learning for Efficiency and Privacy Preservation

This repository contains the source code and experiments for the paper **"Enhancing Tomato Leaf Disease Detection Using Federated Learning for Efficiency and Privacy Preservation."** It includes the design and evaluation of the lightweight **SimpleNetAugDR-3** CNN and its deployment in a Federated Learning (FedAvg) setting for privacy-preserving, edge-oriented tomato leaf disease classification.

---

## Overview

The project addresses tomato leaf disease classification across **10 classes** (9 diseases + healthy) under three goals: high classification accuracy, real-time inference on resource-constrained edge devices, and data-privacy preservation.

Key contributions reproduced here:
- A custom, hardware-aware minimalist CNN (**SimpleNetAugDR-3**) with a compact 2.61 MB footprint and sub-millisecond inference.
- A comparison against scratch-built and pretrained baselines (MobileNet, EfficientNetB0, ResNet50/101, VGG-19, DenseNet121, InceptionV3).
- A **FedAvg** federated learning framework over **4 clients** reaching **>92%** global accuracy.
- Convergence and **Non-IID** (Dirichlet α) robustness studies.
- A feature-level analysis (Grad-CAM, t-SNE, cosine similarity) of the most-confused class (Septoria Leaf Spot).

---

## Model architecture — SimpleNetAugDR-3

A deliberately shallow, hardware-aware CNN: three standard convolutional blocks followed by a single dropout-regularised classification head.

```
Input  : 128 x 128 x 3        (64 x 64 x 3 for the 64-px experiments)
Conv2D : 64 filters, 4 x 4, ReLU   ->  MaxPooling 2 x 2
Conv2D : 64 filters, 4 x 4, ReLU   ->  MaxPooling 2 x 2
Conv2D : 64 filters, 3 x 3, ReLU   ->  MaxPooling 2 x 2
Flatten
Dropout: 0.5
Dense  : 10, softmax
```

- **Optimizer:** Adam (learning rate 1e-3) — **Loss:** categorical cross-entropy
- **Training protocol:** batch size 16, up to 100 epochs, early stopping (patience 5) on the validation loss with best-weight restoration
- **Footprint:** 2.61 MB on disk. The exact parameter count is printed by `create_model().summary()` in each notebook.

The same architecture is used as the local/global model in all federated experiments.

---

## Repository structure

All experiments are provided as self-contained Jupyter notebooks, numbered in a suggested reading order. Each notebook loads the dataset directly (see **Data** below) and writes its tables/figures to an `*_outputs/` folder.

```
.
├── README.md
├── requirements.txt
├── 01_models_benchmark.ipynb         # Scratch + pretrained CNNs @128 and @64   -> Tables II, IV (Figs 4-7)
├── 02_hyperparameter_ablation.ipynb  # LR / batch / optimizer ablation           -> Table III
├── 03_class4_analysis.ipynb          # Grad-CAM, t-SNE, cosine similarity         -> Figs 8, 9; Table V
├── 04_federated_fedavg.ipynb         # FedAvg training + classification report    -> Tables I, VI, VIII
├── 05_fl_convergence_ablation.ipynb  # 20-round convergence study                 -> Table VII, Fig 16
└── 06_noniid_experiments.ipynb       # IID vs Non-IID (alpha=0.5, 0.1)            -> Figs 10-15
```

---

## Notebook → paper results mapping

| Notebook | Produces | Paper reference |
|---|---|---|
| `01_models_benchmark.ipynb` | All CNNs @128×128 (Table II) and best models @64×64 (Table IV); optional per-model confusion matrix / ROC / training curves | **Tables II, IV** (Figs 4–7) |
| `02_hyperparameter_ablation.ipynb` | One-factor-at-a-time LR / batch / optimizer ablation (validation), with a held-out-test confirmation of the selected configuration | **Table III** |
| `03_class4_analysis.ipynb` | Class-4 (Septoria) confusion profile, Grad-CAM, cross-class Grad-CAM, occlusion/saliency, t-SNE, cosine-similarity, colour analysis | **Figs 8, 9; Table V** |
| `04_federated_fedavg.ipynb` | Per-client class distribution, global-model metrics per round (validation), final classification report (held-out test) | **Tables I, VI, VIII** |
| `05_fl_convergence_ablation.ipynb` | 20-round convergence ablation + adopted-round held-out-test confirmation | **Table VII, Fig 16** |
| `06_noniid_experiments.ipynb` | IID / Non-IID Mild (α=0.5) / Severe (α=0.1) scenarios; per-class F1 on the held-out test set | **Figs 10–15** |
| Comparison with related work | — | **Table IX** |

---

## Evaluation protocol (train / validation / test)

All notebooks use a **proper three-way split** so that model selection never touches the data used for final reporting:

- **`train/`** is split (stratified by class) into a **training** set and a **validation** set (default 80/20).
- **`valid/`** is kept as a **held-out test set** — it is never used for training or model selection.
- Training is monitored on the **validation** set (early stopping / round-by-round monitoring); the **final reported metrics** are computed on the **held-out test set**.

For the federated experiments (`04`–`06`):
- The training pool is partitioned across **4 clients** (stratified = IID; Dirichlet α = Non-IID).
- Each round, clients train locally and the server aggregates with **weighted FedAvg**, `w_{t+1} = Σ (n_k / N) w_k`.
- The global model is monitored on the **validation** set each round; final/per-class results are reported on the **held-out test set**.

> Hyperparameter and round-count *selection* (notebooks `02` and `05`) is performed on the **validation** set; the adopted choice is then confirmed once on the held-out test set.

---

## Data

This study uses the **PlantVillage** dataset, specifically the augmented version released by Thonglek et al. (88,156 images), from which the **tomato subset (10 classes)** is used. The notebooks expect the data already organised into `train/` and `valid/` folders with one subfolder per class (the layout of the public *New Plant Diseases Dataset (Augmented)* distribution).

### Classes

| Index | Class | Folder name |
|---|---|---|
| 0 | Bacterial spot | `Tomato___Bacterial_spot` |
| 1 | Early blight | `Tomato___Early_blight` |
| 2 | Late blight | `Tomato___Late_blight` |
| 3 | Leaf mold | `Tomato___Leaf_Mold` |
| 4 | Septoria leaf spot | `Tomato___Septoria_leaf_spot` |
| 5 | Spider mites (two-spotted) | `Tomato___Spider_mites Two-spotted_spider_mite` |
| 6 | Target spot | `Tomato___Target_Spot` |
| 7 | Yellow leaf curl virus | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| 8 | Mosaic virus | `Tomato___Tomato_mosaic_virus` |
| 9 | Healthy | `Tomato___healthy` |

### Expected directory layout

```
new-plant-diseases-dataset/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   └── ... (10 tomato class folders)
└── valid/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    └── ... (10 tomato class folders)
```

The notebooks load **only** the folders whose name contains `Tomato__`, so non-tomato classes in the same archive are ignored automatically.

### Dataset links

- **Dataset used (augmented PlantVillage, `train/` + `valid/` layout):** New Plant Diseases Dataset — Kaggle
  <https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>
  (~87K RGB images, 38 classes at 256×256, pre-split into `train/` and `valid/`; the notebooks use the 10 tomato classes only.)
- **Original PlantVillage source:** <https://github.com/spMohanty/PlantVillage-Dataset>
  (Kaggle mirror: <https://www.kaggle.com/datasets/emmarex/plantdisease>)

### How to obtain the data

1. Download the dataset from the Kaggle link above.
2. Extract it so that the `train/` and `valid/` folders sit under a directory named `new-plant-diseases-dataset/` (see layout above), placed next to the notebooks.
3. If your data lives elsewhere, edit the **path variables at the top of each notebook** — `TRAIN_DIR` / `VALID_DIR` (notebooks `01`, `04`, `05`) or `tomato_train` / `tomato_valid` (notebooks `02`, `03`, `06`).

> **Note:** The raw dataset is **not** committed to this repository due to size and licensing. Availability may be subject to the original providers' usage terms. Please refer to the original source for license details.

---

## Installation

```bash
git clone ⟨your-repo-url⟩
cd ⟨repo-name⟩
pip install -r requirements.txt
```

Recommended: use a virtual environment (`venv` or `conda`).

### Requirements

See `requirements.txt`. Core dependencies:
- Python 3.10+
- TensorFlow / Keras (≥ 2.12)
- NumPy, pandas, scikit-learn
- Matplotlib, seaborn
- opencv-python, Pillow, tqdm

---

## Usage

Each notebook is independent and can be run on its own once the dataset paths are set.

1. Set the dataset path variables at the top of the notebook (see **Data**).
2. For reliable reproduction, run via **Kernel → Restart & Run All** to avoid out-of-order cell state.

```bash
jupyter notebook   # or: jupyter lab
```

Outputs (CSV tables, figures, saved models) are written to a per-notebook `*_outputs/` directory.

> **Reproducibility note.** These notebooks implement a strict train/validation/test separation (final metrics on a held-out test set). Numbers may therefore differ slightly from values obtained under a single-set evaluation; we recommend reporting the held-out-test results produced here.

---

## Hardware

All experiments in the paper were conducted on an **Intel(R) Xeon(R) CPU @ 2.20 GHz with 30 GB RAM**. No GPU is required; the lightweight design targets CPU/edge deployment. Reported inference times correspond to this setup. (A GPU will substantially speed up training of the larger pretrained baselines in `01`.)

---

## Key results

| Model | Accuracy | Inference time | Size |
|---|---|---|---|
| **SimpleNetAugDR-3 (proposed)** | **92.28%** | **0.672 ms** | **2.61 MB** |
| Federated global model (FedAvg, 4 clients) | >92% | — | 2.61 MB |
| MobileNet2 | 90.77% | 0.737 ms | 15.6 MB |
| EfficientNetB0 | 92.98% | 3.184 ms | 51.22 MB |
| ResNet50 | 97.51% | 3.490 ms | 298.05 MB |

The proposed model offers the best trade-off between accuracy, inference speed, and model size among all evaluated architectures.

---

## Authors

⟨Ons Loukil, Slim Amri, Walid Barhoumi⟩

Corresponding author: ⟨walid.barhoumi@enicarthage.rnu.tn⟩
