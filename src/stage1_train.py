import logging
import pandas as pd
import numpy as np
import time
import itertools

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.base import clone
from src.utils import time_convert
from src.config import models, augmented_models

logger = logging.getLogger(__name__)

def stage1_train(X_train: pd.DataFrame, X_train_augmented: pd.DataFrame, y_train: pd.Series, umap_cols: list[str]) -> tuple[dict, dict, dict]:
    """
    Trains Stage 1 base models using a 5-fold cross-validation scheme.

    Args:
        X_train (pd.DataFrame): Original training features.
        X_train_augmented (pd.DataFrame): Training features augmented with UMAP.
        y_train (pd.Series): Training labels.
        umap_cols (list[str]): Names of the UMAP columns.

    Returns:
        tuple: sample_predictions, trained_models, trained_augmented_models
    """
    logger.info("Splitting data into chunks for Stage 1 training...")
    X_train_split = np.array_split(X_train, 5)
    y_train_split = np.array_split(y_train, 5)
    X_train_augmented_split = np.array_split(X_train_augmented, 5)

    features = X_train.columns
    features_augmented = X_train.columns.append(pd.Index(umap_cols))

    trained_models = {}
    trained_augmented_models = {}

    all_model_names = list(models.keys()) + list(augmented_models.keys())
    sample_predictions = {name: {idx: [] for idx in X_train.index} for name in all_model_names} 

    for i in itertools.combinations(range(5), 3):
        logger.info(f"Training on chunks: {i}")
        test_index = list(set(range(5)) - set(i))

        X_train_inner = pd.concat([X_train_split[j] for j in i])
        X_train_inner.columns = features
        X_test_inner = pd.concat([X_train_split[j] for j in test_index])
        X_test_inner.columns = features

        X_train_augmented_inner = pd.concat([X_train_augmented_split[j] for j in i])
        X_train_augmented_inner.columns = features_augmented
        X_test_augmented_inner = pd.concat([X_train_augmented_split[j] for j in test_index])
        X_test_augmented_inner.columns = features_augmented

        y_train_inner = pd.concat([y_train_split[j] for j in i])
        
        # Train standard models
        for name, model_template in models.items():
            model = clone(model_template)
            start_time = time.perf_counter()
            model.fit(X_train_inner, y_train_inner)
            elapsed = time_convert(time.perf_counter() - start_time)
            logger.info(f"Trained {name} in {elapsed}")

            trained_models[name+"_".join(map(str, i))] = model
            probs = model.predict_proba(X_test_inner)

            for idx_pos, sample_idx in enumerate(X_test_inner.index):
                sample_predictions[name][sample_idx].append(probs[idx_pos]) 
        
        # Train augmented models
        for name, model_template in augmented_models.items():
            model = clone(model_template)
            start_time = time.perf_counter()
            model.fit(X_train_augmented_inner, y_train_inner)
            elapsed = time_convert(time.perf_counter() - start_time)
            logger.info(f"Trained {name} in {elapsed}")
            
            trained_augmented_models[name+"_".join(map(str, i))] = model
            probs = model.predict_proba(X_test_augmented_inner)
            
            for idx_pos, sample_idx in enumerate(X_test_augmented_inner.index):
                sample_predictions[name][sample_idx].append(probs[idx_pos])

    return sample_predictions, trained_models, trained_augmented_models