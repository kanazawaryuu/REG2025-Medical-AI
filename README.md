---

## 📂 Repository Structure

```text
├── run_stage1_v4_pipeline.py     # Stage 1: Asynchronous WSI Tiling, QC & GigaPath Feature Extraction
├── run_stage2_parse_labels.py    # Stage 2: NLP Parsing of raw JSON reports to multi-label CSV
├── run_stage2_merge_data.py      # Stage 2.5: Feature-Label Merging & Stratified K-Fold Splitting
├── train_stage3_attention.py     # Stage 3: Training the Multi-Task Gated Attention MIL Model
├── run_stage4_generation.py      # Stage 4: 5-Fold Ensemble Inference & Clinical Report Generation
└── official_eval.py              # Stage 5: Comprehensive Clinical NLP Evaluation (BioBERT & SciSpacy)
```

---

## 💻 Hardware & Environment

To fully reproduce the preprocessing and training pipeline, the following hardware setup is recommended:

- **OS:** Ubuntu 22.04 LTS (or compatible Linux distribution)
- **GPU:** NVIDIA GPU with at least **24 GB VRAM** (e.g., RTX 3090 / RTX 4090 / RTX A5000) for GigaPath feature extraction
- **RAM:** 64 GB or higher recommended for handling gigapixel Whole Slide Images
- **Storage:** High-speed NVMe SSD is strongly recommended for WSI I/O operations

---

## 🚀 Getting Started

### 1. Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision timm openslide-python pandas scikit-learn transformers spacy
python -m spacy download en_core_sci_lg
```

### 2. Data Preparation

Before running the pipeline, please ensure that your raw WSI files (`*.tif` / `*.svs`) and the corresponding ground-truth report file (`train.json`) are placed in your designated `wsi_input_dir`.

> **Note**
>
> Due to medical data privacy regulations, the original dataset is **not** included in this repository.

### 3. Execution Flow

Run the following scripts sequentially to reproduce the complete pipeline:

```bash
# Step 1: Extract WSI Features
# (Make sure 'prov-gigapath' is properly installed and accessible)
python run_stage1_v4_pipeline.py

# Step 2: Parse Clinical Reports & Extract Labels
python run_stage2_parse_labels.py \
    --json_path train.json \
    --output_path breast_cancer_targets.csv

# Step 3: Merge Features & Labels / Create Stratified Splits
python run_stage2_merge_data.py \
    --label_path breast_cancer_targets.csv \
    --features_dir <your_feature_dir>

# Step 4: Train the Multi-Task Gated Attention MIL Model (5-Fold CV)
python train_stage3_attention.py \
    --csv_path final_train_list_multilabel.csv \
    --epochs 20

# Step 5: Ensemble Inference & Clinical Report Generation
python run_stage4_generation.py \
    --model_dir train_stage3_attention \
    --output_path submission.json

# Step 6: Clinical-Grade NLP Evaluation
python official_eval.py
```

---

## 🙏 Acknowledgements

Special thanks to the developers of **prov-gigapath** for providing the foundation models for computational pathology.

We also sincerely thank the open-source communities behind **SciSpacy** and **BioBERT**, whose excellent work made robust medical text evaluation possible.

---

## ✉️ Author

**Jin Zelong**

- **Affiliation:** Master's Student, Information Engineering, Tokyo Polytechnic University
- **Research Interests:** Deep Learning, Computational Pathology, Medical AI, and Multimodal Foundation Models
- **Academic Highlight:** Recipient of the **Best Paper Award** at **ICEIC 2026** for research on attention-based MIL models for breast cancer pathology

If you have any questions about the implementation or are interested in potential research collaborations, please feel free to open a GitHub Issue or contact me via email.
