This repository contains the source code and methodology for developing a multi-class stacking classifier for 13 cancer types using circulating microRNA (miRNA) expression profiles from public microarray datasets.

## Repository Structure

```text
TISSUE_OF_ORIGIN/
├── data/                       # Raw CSV files and generated joblib artifacts
│   ├── dat.csv
│   ├── sample_predictions.joblib
│   ├── samples.csv
│   ├── train_test_split.joblib
│   ├── umap_data.joblib
│   └── X_train_stage2.joblib
├── models/                     # Saved joblib models and keys
│   ├── all_trained_models.joblib
│   ├── metaclassifier.joblib
│   └── model_keys.joblib
├── scripts/                    # Runnable pipeline entry points
│   ├── main_pipeline.py        # End-to-end training and evaluation script
│   ├── train.py                # Decoupled training execution script
│   └── validate.py             # Decoupled validation/inference execution script
├── src/                        # Core package source code
│   ├── config.py               # Centralized configuration, paths, and model configs
│   ├── data_prep.py            # Data loading and stratified splitting logic
│   ├── stage1_train.py         # Base learner CV and probability aggregation
│   ├── stage2_train.py         # Metaclassifier training logic
│   ├── umap_aug.py             # UMAP reduction and feature augmentation
│   ├── utils.py                # Time conversion and formatting utilities
│   └── validation.py           # Evaluation functions and metrics calculations
├── .gitignore
├── documentation.md            # Raw methodology notes
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies

INSTALLATION

pip install -r requirements.txt  

SCRIPTS

python -m scripts.train  
python -m scripts.validate 