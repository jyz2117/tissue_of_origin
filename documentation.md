```markdown
# Methodology

## 1. Study Design

This project develops a multi-class stacking classifier for 13 cancer types using circulating microRNA (miRNA) expression profiles from public microarray datasets[cite: 9].

## 2. Data Sources

**GSE211692** - Serum microRNA profiles of 9,921 patients with 13 types of human solid cancers[cite: 9]:

| Cancer Type | Sample Count |
| :--- | :--- |
| Lung cancer | 1699 |
| Colorectal cancer | 1596 |
| Gastric cancer | 1418 |
| Prostate cancer | 1027 |
| Pancreatic cancer | 851 |
| Breast cancer | 675 |
| Esophageal cancer | 566 |
| Biliary tract cancer | 402 |
| Ovarian cancer | 400 |
| Bladder cancer | 399 |
| Liver cancer | 348 |
| Sarcoma | 299 |
| Glioma | 241 |

## 3. Data Processing

* Label encoder applied for compatibility with XGBoost 3.3.0[cite: 9].

### `train_test_split`

* Stratified by `disease.state`, `age_group` (cut from age), and `Sex`[cite: 9].
* Split 4:1 for outer train and test sets[cite: 9].

```python
>>> X_train.shape
(7936, 2565)
>>> X_test.shape
(1985, 2565)

```

### UMAP

* Trained UMAP on the outer train set created above and collected 3 feature columns.


* Transformed the test set using the UMAP fitted to the train set and collected 3 feature columns.


* Created `X_train_augmented` and `X_test_augmented` with the 3 feature columns concatenated to original features.



## 4. Model Training

### Stage 1

#### Cross-Validation (CV)

**Singular fold execution:**

* The outer train set (`X_train` and `X_train_augmented`) was split into 5 chunks.


* 3 chunks were allocated to an inner train set and 2 chunks were allocated to an inner test set.


* Several learners were trained on the inner train set:


* **Trained on raw data:** `['RandomForest', 'ExtraTrees', 'GLM_Logistic', 'SVM_Linear']`

* **Trained on augmented data (UMAP):** `['XGB', 'MLP1', 'MLP2']`



* Class probabilities were predicted for the remaining samples in the inner test set and collected.



**Aggregation across chunk combinations:**
This process was repeated across all 10 combinations of chunks. In total:

* Each model was trained 10 times (10 separate models), leaving 70 total models trained:


```python
>>> len(trained_models.keys())
40
>>> len(trained_augmented_models.keys())
30

```


* Each sample appeared in 4 inner test sets and thus had 4 predicted probability arrays per model type.


* Each sample's 4 sets of predicted probabilities were geometrically averaged and concatenated to a new dataframe:


* **Rows:** 7936 (patients)


* **Cols:** 91 (7 models × 13 classes)





### Stage 2

* A metaclassifier (XGBoost) was trained on the aggregated output from Stage 1.



## 5. Model Evaluation

* Stage 1 models trained previously were applied to the outer test set (unseen data).


* Predicted probabilities were averaged across all 10 variants of each model (rather than only 4).


* Results were concatenated and fed into the metaclassifier for final predictions.



## 6. Key Results

*(To be populated)*

```

```