# Cancer miRNA Stacking Classifier

This repository contains the source code and methodology for developing a multi-class stacking classifier for 13 cancer types using circulating microRNA (miRNA) expression profiles from public microarray datasets[cite: 9].

---

## Repository Structure

```text
project_root/
├── data/                   # Contains raw CSV files and generated joblib data
├── models/                 # Contains saved joblib models (base models and metaclassifier)
├── scripts/
│   └── main_pipeline.py    # Main entry point for executing the training and evaluation pipeline
├── src/                    # Source code package
│   ├── __init__.py
│   ├── config.py           # Configuration variables, paths, and model definitions
│   ├── data_prep.py        # Data loading and stratified outer splitting
│   ├── umap_aug.py         # UMAP dimensionality reduction and augmentation
│   ├── stage1_train.py     # Base model training via custom 5-fold CV
│   ├── stage2_train.py     # Metaclassifier training
│   ├── validation.py       # Test set evaluation and scoring
│   └── utils.py            # Helper functions
├── README.md               # Project documentation
└── requirements.txt        # package versions



pip install -r requirements.txt