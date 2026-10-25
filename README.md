# Multi-Camera Person Re-Identification on MEVID: A Comparative Study of Deep CNN Architectures Under Clothing Change

An empirical research project for **42177 Image Processing and Pattern Recognition (Team 8, UTS)** investigating video-based Person Re-Identification (ReID) across non-overlapping cameras, indoor/outdoor lighting transitions, and multi-day clothing changes on the **MEVID (Multi-View Extended Videos with Identities)** benchmark.

---

## 1. Research Overview & Objectives

While deep Person Re-Identification models achieve near-saturated accuracy (>94% Rank-1) on short-term benchmarks such as Market-1501, those datasets assume every pedestrian wears the exact same outfit across all cameras. In real-world multi-day surveillance, clothing changes cause appearance-based descriptors to fail dramatically.

This repository provides a standardized training and multi-protocol evaluation framework built on **PyTorch** and **FastReID** to benchmark and compare **three deep ReID architectures** under identical evaluation protocols:

1. **BoT-ResNet50** — Global ResNet-50 baseline trained with the *Bag of Tricks* recipe (`BNNeck`, `Last Stride = 1`, `Label Smoothing`, `Batch-Hard Triplet Loss`, `Random Erasing`) initialized from ImageNet weights.
2. **AGW-ResNet50-IBN** — Enhanced ResNet-50 backbone incorporating **Instance-Batch Normalization (`IBN-a`)**, **Non-Local Attention** blocks, **Generalized Mean (`GeM`) Pooling**, and **Weighted Regularization Triplet (`WRT`) Loss**, fine-tuned from **MSMT17** pre-trained weights.
3. **OSNet + MGN** — Lightweight omni-scale feature learning backbone (**OSNet-x1.0**) initialized from **MSMT17** weights and extended with a **Multi-Granularity Network (`MGN`)** horizontal-stripe part head (1,792-D combined global + local embedding).

---

## 2. Evaluation Protocols

Using MEVID's explicit identity (`PID`), outfit (`OID`), and camera (`CID`) annotations, every model is evaluated across three official protocols via GPU-accelerated cosine similarity ranking (`src/data/mevid_evaluator.py`):

- **General Protocol:** Standard cross-camera ReID where all true identity matches from different cameras are valid positives regardless of clothing.
- **Same-Outfit Protocol:** Isolates cross-camera gallery matches where the subject wears the **same outfit** as the query (different-outfit matches are masked out).
- **Cross-Outfit Protocol:** Isolates cross-camera gallery matches where the subject wears a **completely different outfit** from the query (same-outfit matches are masked out), directly measuring clothing-invariant biometric recognition.

---

## 3. Experimental Results Summary

### Table 1: Cross-Model Performance Comparison on MEVID (Base vs. Fine-Tuned)

| Model Architecture | Pre-Training | General Rank-1 (Gain) | General mAP (Gain) | Same-Outfit Rank-1 (Gain) | Same-Outfit mAP (Gain) | Cross-Outfit Rank-1 (Gain) | Cross-Outfit mAP (Gain) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoT-ResNet50** | ImageNet | `33.45%` (+3.40) | `12.89%` (+2.47) | `36.36%` (+3.65) | `16.90%` (+3.95) | `1.42%` (0.00) | `1.52%` (−0.07) |
| **AGW-ResNet50-IBN** | MSMT17 | `37.14%` (+7.70) | `15.46%` (+3.99) | `40.28%` (+8.33) | `18.34%` (+4.58) | `2.84%` (+1.58) | `2.68%` (+1.25) |
| **OSNet + MGN** | MSMT17 | **`45.89%`** (+4.76) | **`17.15%`** (+3.12) | **`51.58%`** (+4.88) | **`21.23%`** (+4.01) | **`4.45%`** (+1.80) | **`2.96%`** (+1.17) |

### Key Research Findings
- **Domain Pre-Training Matters:** Initializing from pedestrian-specific **MSMT17** weights (`AGW-ResNet50-IBN` and `OSNet + MGN`) with initial backbone freezing outperformed generic ImageNet initialization (`BoT-ResNet50`) across all protocols.
- **Attention & Multi-Granularity Excels on Same-Outfit Retrieval:** `IBN-a`, `Non-Local Attention`, and `GeM` pooling (`AGW-ResNet50-IBN`) boosted Same-Outfit Rank-1 to `40.28%` (+8.33 pp), while horizontal part partitioning (`OSNet + MGN`) reached `51.58%` Rank-1 (`66.51%` Rank-10).
- **The Clothing-Bias Bottleneck:** All three 2D RGB architectures lost **>91% of their Rank-1 accuracy** when moving from Same-Outfit (`36.36%–51.58%`) to Cross-Outfit (`1.42%–4.45%`), demonstrating that standard aggregate ReID scores (`General Protocol`) mask severe shortcut learning on clothing color and texture.

---

## 4. Repository Structure

```text
MultiCameraTracking/
├── configs/
│   ├── Base-bagtricks.yml          # Base FastReID Bag-of-Tricks configuration
│   ├── mevid_resnet50.yml          # Model 1: BoT-ResNet50 configuration
│   ├── mevid_agw_r50_ibn.yml       # Model 2: AGW-ResNet50-IBN configuration
│   └── mevid_osnet.yml             # Model 3: OSNet-x1.0 configuration
├── src/
│   ├── data/
│   │   ├── fastreid_mevid.py       # Custom FastReID dataset registry for MEVID
│   │   ├── mevid_dataset.py        # Tracklet & frame parser ([PID]O[OID]C[CID]...)
│   │   └── mevid_evaluator.py      # GPU-accelerated 3-protocol CMC & mAP evaluator
│   └── models/                     # Backbone wrappers & feature extraction utilities
├── scripts/
│   ├── download_mevid.sh           # Automated MEVID dataset downloader & organizer
│   ├── train.py                    # Unified FastReID training entry point
│   ├── evaluate.py                 # Standard model checkpoint evaluation script
│   ├── eval_mevid_protocols.py     # Multi-protocol (General / Same / Cross) evaluation
│   ├── agw_resnet_ibn_25740743.ipynb # AGW-ResNet50-IBN training & evaluation notebook
│   └── generate_report_plots.py    # Generates loss curves & protocol evaluation plots
├── results/
│   ├── loss_curves.png             # 22-epoch training loss dynamics (AGW-ResNet50-IBN)
│   ├── accuracy_curves.png         # Validation Rank-1/5/10 & mAP progression
│   └── mevid_protocol_comparison_chart.png # Same-Outfit vs. Cross-Outfit comparison
├── requirements.txt                # Core Python dependencies
└── requirements-train.txt          # Training & FastReID dependencies
```

---

## 5. Installation & Dataset Setup

### 5.1 Environment Setup
```bash
# Install core and training dependencies
pip install -r requirements.txt
pip install -r requirements-train.txt
```

### 5.2 Download & Prepare the MEVID Dataset
Run the automated download script to fetch the MEVID dataset splits and organize them under `data/mevid/`:
```bash
bash scripts/download_mevid.sh
```

---

## 6. Usage: Training & Multi-Protocol Evaluation

### 6.1 Training Candidate Models
Train any of the three candidate architectures using `scripts/train.py` and the corresponding YAML configuration in `configs/`:

```bash
# 1. Train AGW-ResNet50-IBN (fine-tuned from MSMT17 weights)
python scripts/train.py --config-file configs/mevid_agw_r50_ibn.yml

# 2. Train BoT-ResNet50 Baseline
python scripts/train.py --config-file configs/mevid_resnet50.yml

# 3. Train OSNet-x1.0
python scripts/train.py --config-file configs/mevid_osnet.yml
```

### 6.2 Running the 3-Protocol Evaluation (General, Same-Outfit, Cross-Outfit)
Evaluate any trained or pre-trained checkpoint across all three MEVID protocols using `scripts/eval_mevid_protocols.py`:

```bash
python scripts/eval_mevid_protocols.py \
    --config-file configs/mevid_agw_r50_ibn.yml \
    --weights weights/agw_r50_ibn_mevid_best.pth
```
