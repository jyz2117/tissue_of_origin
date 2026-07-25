# Methodology

## 1. Study Design

This project develops a multi-class stacking classifier for 13 cancer types using circulating microRNA (miRNA) expression profiles from public microarray datasets. 

## 2. Data Sources

GSE211692- Serum microRNA profiles of 9,921 patients with 13 types of human solid cancers:

lung cancer             1699
colorectal cancer       1596
gastric cancer          1418
prostate cancer         1027
pancreatic cancer        851
breast cancer            675
esophageal cancer        566
biliary tract cancer     402
ovarian cancer           400
bladder cancer           399
liver cancer             348
sarcoma                  299
glioma                   241

## 3. Data Processing

Label encoder for compatability with XGboost 3.3.0

### train_test_split

- stratified by disease.state, age_group (cut from age), and sex
- split 4:1 for outer train and test

>>> X_train.shape
(7936, 2565)
>>> X_test.shape
(1985, 2565)

### UMAP

- trained UMAP on outer train set created above and collected 3 feature columns
- transformed test set using the UMAP fitted to train set and collected 3 feature columns
- created X_train_augmented and X_test_augmented with the 3 feature columns concatenated original features

## 4. Model Training

### Stage 1

#### CV

Singular fold:

    - the outer train set (X_train and X_train_augmented) was split into 5 chunks
    - 3 chunks were allocated to an inner train set and 2 chunks were allocated to an inner test set
    - several learners were trained on the inner train set as described below:

    Trained on raw data:                ['RandomForest', 'ExtraTrees', 'GLM_Logistic', 'SVM_Linear']
    Trained on augmented data (UMAP):   ['XGB', 'MLP1', 'MLP2']

    - class probabilities were predicted for the remaining samples in the inner test set and collected

This process was repeated across all 10 combinations of chunks. In total,

    - each model was trained 10 times (10 separate models), leaving 70 total models trained

        >>> len(trained_models.keys())
        40
        >>> len(trained_augmented_models.keys())
        30

    - each sample appeared in 4 inner test sets and thus had 4 predicted probabilities for a given type of model
    - each sample's 4 sets of predicted probabilities were geometrically averaged and concatentated to a new dataframe:

        rows: 7936 (patients)
        cols: 91 (7 models * 13 classes)

### stage 2

A metaclassifier (XGB) was trained on the output from stage 1

## 5. Model Evaluation

- Stage 1 models trained previously were now applied to the outer test set (unseen)
- Note that predicted probabilities were averaged across all 10 variants of each model (rather than only 4)
- Results were concatenated and fed into the metaclassifier for predictions

## 6. Key Results
