import pandas as pd
import numpy as np


from scipy.stats import gmean
from sklearn.metrics import accuracy_score

import joblib
from pathlib import Path

# load data
X_train, X_test, y_train, y_test = joblib.load(Path("data") / "train_test_split.joblib")
X_train_augmented, X_test_augmented, umap_cols = joblib.load(Path("data") / "umap_data.joblib")

# load models
models, augmented_models = joblib.load(Path("models") / "model_keys.joblib")
trained_models = joblib.load(Path("models") / "trained_models.joblib")
trained_augmented_models = joblib.load(Path("models") / "trained_augmented_models.joblib")
metaclassifier = joblib.load(Path("models") / "metaclassifier.joblib")


### stage 1 predictions

final_stage1_test_features = []

for name in models.keys():
    model_fold_probs = [] # will be a list of prediction arrays, PER MODEL
    
    # gets models by looking for model name in the keys
    specific_trained_models = [m for k, m in trained_models.items() if k.startswith(name)]
    
    for model in specific_trained_models:
        probs = model.predict_proba(X_test)
        model_fold_probs.append(probs) # add to 
        
    stacked_probs = np.stack(model_fold_probs) # stack so we can gmean
    aggregated_prob = gmean(stacked_probs, axis=0) # gmean only for CURRENT model, which will return a 2d pred array where rows = samples, col = 13
    
    df_model_probs = pd.DataFrame(
        aggregated_prob, 
        index=X_test.index,
        columns=[f"{name}_Class{c}" for c in range(aggregated_prob.shape[1])]
    ) # convert to dataframe, samples x 13

    final_stage1_test_features.append(df_model_probs) # add to list- e.g. item 1 will be the averaged probabilities for X_test as predictted by only ONE model

for name in augmented_models.keys():
    model_fold_probs = []
    
    specific_trained_models = [m for k, m in trained_augmented_models.items() if k.startswith(name)]
    
    for model in specific_trained_models:
        probs = model.predict_proba(X_test_augmented)
        model_fold_probs.append(probs)
        
    stacked_probs = np.stack(model_fold_probs)
    aggregated_prob = gmean(stacked_probs, axis=0)
    
    df_model_probs = pd.DataFrame(
        aggregated_prob, 
        index=X_test_augmented.index,
        columns=[f"{name}_Class{c}" for c in range(aggregated_prob.shape[1])]
    )

    final_stage1_test_features.append(df_model_probs)

X_test_stage2 = pd.concat(final_stage1_test_features, axis=1) # turn into one big wide dataframe
X_test_stage2 = pd.concat([X_test_augmented, X_test_stage2], axis=1) # add original and augmented data

### stage 2 predictions

preds = metaclassifier.predict(X_test_stage2)

accuracy_score(y_test, preds) # 0.8896725440806046, 0.8720403022670025 for just ranger and mlp1

### print individual model performance

for name, model in trained_models.items():
    print(name.ljust(26) + str(model.score(X_test, y_test)))

for name, model in trained_augmented_models.items():
    print(name.ljust(26) + str(model.score(X_test_augmented, y_test)))