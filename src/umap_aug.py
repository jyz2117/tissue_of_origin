import logging
import pandas as pd
import umap
import joblib

from sklearnex import patch_sklearn
patch_sklearn()

from src.config import umap_cols, MODELS_DIR

logger = logging.getLogger(__name__)

def fit_umap_train(X_train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Trains a UMAP reducer on the training set, appends components, and saves the reducer.

    Args:
        X_train (pd.DataFrame): Training feature matrix.

    Returns:
        tuple: Augmented X_train and list of UMAP column names.
    """
    logger.info("Initializing and fitting UMAP...")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42, n_jobs=1)

    X_train_umap = umap_reducer.fit_transform(X_train)
    X_train_augmented = X_train.copy()
    X_train_augmented[umap_cols] = pd.DataFrame(X_train_umap, index=X_train.index)

    # Save the fitted UMAP reducer for inference
    joblib.dump(umap_reducer, MODELS_DIR / "umap_reducer.joblib")

    return X_train_augmented, umap_cols


def transform_umap_test(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the saved UMAP reducer and transforms the test set.

    Args:
        X_test (pd.DataFrame): Testing feature matrix.

    Returns:
        pd.DataFrame: Augmented X_test.
    """
    logger.info("Loading UMAP reducer and transforming test set...")
    umap_reducer = joblib.load(MODELS_DIR / "umap_reducer.joblib")
    
    X_test_umap = umap_reducer.transform(X_test)
    X_test_augmented = X_test.copy()
    X_test_augmented[umap_cols] = pd.DataFrame(X_test_umap, index=X_test.index)

    return X_test_augmented