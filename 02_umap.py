import pandas as pd
import numpy as np
import time
import umap

from sklearnex import patch_sklearn # for performance
patch_sklearn()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import joblib

def time_convert(time):
    if time < 60:
        return f"{time:.2f} seconds"
    elif time < 3600:
        return f"{time/60:.2f} minutes"
    else:
        return f"{time/3600:.2f} hours"

            ############
            ### UMAP ###
            ############

X_train, X_test, y_train, y_test = joblib.load("train_test_split.joblib")

# UMAP
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42, n_jobs=1)

start_time = time.perf_counter()
X_train_umap = umap_reducer.fit_transform(X_train) # train umap only on X_train, create its output features
end_time = time.perf_counter()
print(f"UMAP time: {time_convert(end_time - start_time)}")

umap_cols = ['A', "B", "C"] # colnames for umap features
X_train_augmented = X_train.copy() # initialize augmented train set
X_train_augmented[umap_cols] = pd.DataFrame(X_train_umap, index=X_train.index) # attach umap features to augmented train set
X_train_umap = pd.DataFrame(X_train_umap, columns=umap_cols) # convert umap features to separate dataframe

X_test_umap = umap_reducer.transform(X_test) # apply umap to test set
X_test_augmented = X_test.copy() # initialize augmented test set
X_test_augmented[umap_cols] = pd.DataFrame(X_test_umap, index=X_test.index) # create augmented data
X_test_umap = pd.DataFrame(X_test_umap, columns=umap_cols) # convert umap features to separate dataframe

joblib.dump((X_train_augmented, X_test_augmented, umap_cols), "umap_data.joblib")