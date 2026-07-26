import pandas as pd
import numpy as np
import time
import itertools

from sklearnex import patch_sklearn # for performance
patch_sklearn()

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import gmean

import joblib
from pathlib import Path


def time_convert(time):
    if time < 60:
        return f"{time:.2f} seconds"
    elif time < 3600:
        return f"{time/60:.2f} minutes"
    else:
        return f"{time/3600:.2f} hours"

X_train, X_test, y_train, y_test = joblib.load(Path("data") / "train_test_split.joblib")
X_train_augmented, X_test_augmented, umap_cols = joblib.load(Path("data") / "umap_data.joblib")

seed = 42
            #####################
            ### DEFINE MODELS ###
            #####################

models = {
        "RandomForest": RandomForestClassifier(random_state=seed), #ranger
        "ExtraTrees": ExtraTreesClassifier(random_state=seed),
        "GLM_Logistic": LogisticRegression(max_iter=5000, random_state=seed),

        "SVM_Linear": CalibratedClassifierCV(
            estimator=LinearSVC(
                    C=50, 
                    penalty = 'l1', 
                    dual=False, 
                    loss='squared_hinge',
                    random_state=seed
                ),
            method='sigmoid', 
            cv=5  # Matches the paper's 5-fold cross-validation scheme[cite: 1, 2]
        )
        
        
        #"KNN": KNeighborsClassifier(n_neighbors=5)
    }

augmented_models = {
        "XGB": XGBClassifier(
            objective='multi:softmax',
            learning_rate=0.05, 
            max_depth=5, 
            n_estimators=1000, 
            subsample=0.5, 
            colsample_bytree=1.0, 
            random_state=seed
        ), #xgb
        "MLP1": MLPClassifier(
            hidden_layer_sizes=(512, 512, 512),  # 3 hidden layers
            activation='relu',                   # ReLU activation function
            solver='adam',                       # Adam optimizer
            max_iter=1000,                        # Max epochs set to 100
            random_state=seed                      # For reproducibility
        ),
        "MLP2": MLPClassifier(
            hidden_layer_sizes=(512, 512, 512, 512),  # 4 hidden layers
            activation='relu',                       # ReLU activation function
            solver='adam',                           # Adam optimizer
            max_iter=1000,                            # Max epochs set to 100
            random_state=seed                          # For reproducibility
        )
    }

joblib.dump((models, augmented_models), Path("models") / "model_keys.joblib")

            ##################################
            ### INNER 3:2 CROSS VALIDATION ###
            ##################################

# splitting into 5 chunks for INNER
X_train_split = np.array_split(X_train, 5)
y_train_split = np.array_split(y_train, 5)

# create augmented inner chunks
X_train_augmented_split = np.array_split(X_train_augmented, 5)

features = X_train.columns # get mir names for later
features_augmented = X_train.columns.append(pd.Index(umap_cols))

trained_models = {}
trained_augmented_models = {}

        ### modify how probabilities are treated

all_model_names = list(models.keys()) + list(augmented_models.keys())
sample_predictions = {name: {idx: [] for idx in X_train.index} for name in all_model_names} 
# initialize nested dictionary: all models, for which each model contains every sample, for which each sample is a list of 1d prob arrays

for i in itertools.combinations(range(5), 3): # this loop gets all combinations
    print(f"train chunks: {i}")
    test_index = list(set(range(5)) - set(i)) # gets combination in 0-4 that is not in i

    X_train_inner = pd.concat([X_train_split[j] for j in i]) # conct 3, j for i since i is a tuple of 3 
    X_train_inner.columns = features
    X_test_inner = pd.concat([X_train_split[j] for j in test_index]) # does the opposite
    X_test_inner.columns = features

    X_train_augmented_inner = pd.concat([X_train_augmented_split[j] for j in i])
    X_train_augmented_inner.columns = features_augmented
    X_test_augmented_inner = pd.concat([X_train_augmented_split[j] for j in test_index])
    X_test_augmented_inner.columns = features_augmented

    y_train_inner = pd.concat([y_train_split[j] for j in i])
    y_test_inner = pd.concat([y_train_split[j] for j in test_index])
    # all inner train test splitting ^^^
    
    for name, model_template in models.items(): # models in models

        model = clone(model_template) # create new instance in memory

        start_time = time.perf_counter()
        model.fit(X_train_inner, y_train_inner) # train
        end_time = time.perf_counter()
        print(f"{name} time: {time_convert(end_time - start_time)}")

        trained_models[name+"_".join(map(str, i))] = model # save model

        probs = model.predict_proba(X_test_inner) # (3174, 13), probabilities generated by only 1 model

        for idx_pos, sample_idx in enumerate(X_test_inner.index):
            sample_predictions[name][sample_idx].append(probs[idx_pos]) 
            # appends 13 class probabilities by idx_pos, which is the index of that sample in X_test_inner
            # keeps order, and [sample_idx] keeps sample name
            # sample_predictions first layer is models, then within each model is every sample from X_test_inner (repeated) with each having 13 class probs (from that model) attached
    
    for name, model_template in augmented_models.items():

        model = clone(model_template) # create new instance in memory

        start_time = time.perf_counter()
        model.fit(X_train_augmented_inner, y_train_inner) # train
        end_time = time.perf_counter()
        print(f"{name} time: {time_convert(end_time - start_time)}")
        
        trained_augmented_models[name+"_".join(map(str, i))] = model # save model

        probs = model.predict_proba(X_test_augmented_inner)
        
        for idx_pos, sample_idx in enumerate(X_test_augmented_inner.index):
            sample_predictions[name][sample_idx].append(probs[idx_pos])


### aggregate probss and create stage 2 training data
models_to_skip = ("MLP2", "GLM_")

final_stage1_features = [] # aggregate 4 predictions per sample using the geometric mean (per model)

for name in all_model_names: # iterate first thru models because we need to gmean probs per model (this is COLUMNS)
    if name.startswith(models_to_skip):
        continue
    model_gmean_probs = []
    
    for sample_idx in X_train.index:
        four_probs = np.array(sample_predictions[name][sample_idx]) # for a single sample
        # 4 rows represending models, 13 cols representing classes
        # 10 combinations of 3:2 split, so we end up with 10 versions of svm trained.
        # there are thus 10 predict_proba arrays, but each sample only appear 4 times (to avoid label leakage), which this loop collects
        
        
        aggregated_prob = gmean(four_probs, axis=0) # gmean across models, returns a 1d array where col = averaged probs
        model_gmean_probs.append(aggregated_prob) # appending like this keeps the sample order, as we iterate thru sample indexes
        #model_gmean_probs is a list that has columns = class probs, rows = samples ONLY for the current model
        
    df_model_probs = pd.DataFrame(
        model_gmean_probs, 
        index=X_train.index,
        columns=[f"{name}_Class{c}" for c in range(len(model_gmean_probs[0]))] # get model names and create column names
    ) # converts to a df that contains class probs for all samples in X_train, for CURRENT model

    final_stage1_features.append(df_model_probs) # adds this to list containing these evaluations for all samples for ALL models

X_train_stage2 = pd.concat(final_stage1_features, axis=1) # concatenate list horizontally into pd
# X_train_stage2 = pd.concat([X_train_augmented, X_train_stage2], axis=1) # add original + umap

joblib.dump(X_train_stage2, Path("data") / "X_train_stage2.joblib")


### SAVE MODEL DICTIONARY

# joblib.dump(trained_models, Path("models") / "trained_models.joblib")
# joblib.dump(trained_augmented_models, Path("models") / "trained_augmented_models.joblib")


### metaclassifier training (XGB)

metaclassifier = XGBClassifier(
    objective='multi:softmax', 
    learning_rate=0.1, 
    max_depth=5, 
    n_estimators=1000,
    subsample=0.9, 
    colsample_bytree=0.7,
    random_state=seed
) #xgb

# metaclassifier = LinearSVC(
#                     C=50, 
#                     penalty = 'l1', 
#                     dual=False, 
#                     loss='squared_hinge',
#                     random_state=seed
#                 )

start_time = time.perf_counter()        
metaclassifier.fit(X_train_stage2, y_train)
end_time = time.perf_counter()
print(f"metaclassifier fit time: {time_convert(end_time - start_time)}")

joblib.dump(metaclassifier, Path("models") / "metaclassifier.joblib")

