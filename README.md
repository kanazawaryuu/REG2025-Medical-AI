# 🔬 REG2025 Medical Image AI Challenge - Official Solution

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![SciSpacy](https://img.shields.io/badge/NLP-SciSpacy-blue?style=for-the-badge)
![BioBERT](https://img.shields.io/badge/Model-BioBERT-orange?style=for-the-badge)

This repository contains the core source code and official solution for the **REG2025 Medical Image AI Challenge**. 

Our approach proposes a robust, end-to-end computational pathology pipeline designed to process gigapixel Whole Slide Images (WSIs) and automatically generate clinical-grade, multi-label diagnostic reports, specifically optimized for breast cancer pathology.

---

## ✨ Core Technical Highlights

* **GigaPath Foundation Model Integration:** Leveraged `prov-gigapath` for state-of-the-art tile-level feature extraction.
* **Gated Attention MIL with Multi-Task Learning:** Engineered a customized Multiple Instance Learning (MIL) architecture with 6 auxiliary Prediction Heads (MLPs) to capture fine-grained sub-features (e.g., NST Grading, DCIS Necrosis).
* **Long-Tail Distribution Optimization:** Successfully mitigated severe class imbalance through Asymmetric Loss (ASL) and strategic Weighted Random Sampling, ensuring high recall on ultra-rare categories (e.g., Metaplastic Carcinoma).
* **Asynchronous CPU/GPU Pipeline:** Designed a highly efficient, dual-thread multiprocessing pipeline for WSI preprocessing (Tiling, QC, and Extraction), maximizing hardware utilization.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, PyTorch Lightning, TIMM
* **WSI Processing:** OpenSlide, OpenCV, PIL
* **NLP & Evaluation:** Transformers (Hugging Face), SciSpacy, BioBERT
* **Data Processing:** Pandas, NumPy, Scikit-learn

---

## 🧬 Pipeline Architecture

The complete workflow is divided into 5 modular stages, ensuring reproducibility and clinical accuracy:

### 🔹 Stage 1: Asynchronous WSI Preprocessing
* Performs multi-threaded virtual downsampling, background filtering (white intensity/saturation), and blur detection.
* Extracts robust feature embeddings utilizing the pre-trained `prov-gigapath` model.

### 🔹 Stage 2: Pathology Text Parsing & Label Engineering
* Employs advanced RegEx-based parsing to extract 24 fundamental pathology classes from raw JSON reports.
* Disentangles complex sub-features (Nottingham Grade for NST, Nuclear Grade/Type for DCIS) for multi-task learning.

### 🔹 Stage 3: Multi-Task MIL Model Training
* Trains the **Gated Attention MIL** network using 5-Fold Cross Validation.
* Integrates Mixed Precision Training (AMP) and dynamic threshold searching to optimize the Macro F1 score across all classes.

### 🔹 Stage 4: 5-Fold Ensemble Inference & Report Generation
* Aggregates predictions from the 5-fold ensemble to guarantee stability.
* Translates logits into accurate, physician-style text reports using confidence-calibrated thresholds.

### 🔹 Stage 5: Clinical-Grade NLP Evaluation
* Evaluates generated reports using a custom composite metric.
* Combines **Semantic Similarity** (BioBERT), **Entity Matching** (SciSpacy), and **Syntactic Metrics** (BLEU-4, ROUGE-L) to strictly simulate real-world clinical tolerance.


## 📂 Repository Structure

```text
├── run_stage1_v4_pipeline.py     # Stage 1: Asynchronous WSI Tiling, QC & GigaPath Feature Extraction
├── run_stage2_parse_labels.py    # Stage 2: NLP Parsing of raw JSON reports to multi-label CSV
├── run_stage2_merge_data.py      # Stage 2.5: Feature-Label merging and Stratified K-Fold splitting
├── train_stage3_attention.py     # Stage 3: Training the Multi-Task Gated Attention MIL model
├── run_stage4_generation.py      # Stage 4: 5-Fold Ensemble Inference & Clinical Report Generation
└── official_eval.py              # Stage 5: Comprehensive clinical NLP evaluation (BioBERT & SciSpacy)


💻 Hardware & Environment
To fully reproduce the preprocessing and training pipeline, the following hardware setup is recommended:
OS: Ubuntu 22.04 LTS (or compatible Linux distribution)
GPU: NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090 / 4090 / A5000) for GigaPath feature extraction.
RAM: 64GB+ recommended for handling gigapixel Whole Slide Images.
Storage: High-speed NVMe SSD is strictly recommended for WSI I/O operations.

🚀 Getting Started
1. Prerequisites
Ensure you have the required dependencies installed:

```Bash
pip install torch torchvision timm openslide-python pandas scikit-learn transformers spacy
python -m spacy download en_core_sci_lg
```

2. Data Preparation
Before running the pipeline, please ensure your raw WSI data (*.tif / *.svs) and the ground truth report file (train.json) are correctly placed in your designated wsi_input_dir. Due to medical data privacy regulations, the original dataset is not included in this repository.


3. Execution Flow
Execute the scripts sequentially to reproduce the end-to-end pipeline:

```Bash
# Step 1: Extract WSI Features (Make sure 'prov-gigapath' is accessible)
python run_stage1_v4_pipeline.py
# Step 2: Parse Clinical Reports & Extract Labels
python run_stage2_parse_labels.py --json_path train.json --output_path breast_cancer_targets.csv
# Step 3: Feature-Label Merging & Stratified Data Splitting
python run_stage2_merge_data.py --label_path breast_cancer_targets.csv --features_dir <your_feature_dir>
# Step 4: Train Multi-Task Gated Attention MIL Model (5-Fold CV)
python train_stage3_attention.py --csv_path final_train_list_multilabel.csv --epochs 20
# Step 5: Ensemble Inference & Text Report Generation
python run_stage4_generation.py --model_dir train_stage3_attention --output_path submission.json
# Step 6: Clinical-Grade NLP Evaluation
python official_eval.py
```

🙏 Acknowledgements
Special thanks to the developers of prov-gigapath for providing the foundational vision transformer models for pathology.
Thanks to the open-source community behind SciSpacy and BioBERT for enabling robust medical text evaluation.

✉️ Author
Jin Zelong

Affiliation: Master's Student, Information Engineering, Tokyo Polytechnic University.
Research Focus: Deep Learning, Computational Pathology, Medical AI & Multi-modal Foundation Models.
Academic Highlight: Awarded the Best Paper Award at the ICEIC 2026 international conference for research on attention-based MIL models for breast cancer pathology.
Feel free to reach out via GitHub Issues or email if you have any questions regarding the implementation, or if you are interested in potential collaborations!
