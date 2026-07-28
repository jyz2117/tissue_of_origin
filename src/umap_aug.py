import logging
import pandas as pd
import umap

from sklearnex import patch_sklearn
patch_sklearn()

from src.config import umap_cols

logger = logging.getLogger(__name__)

def umap_aug(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Trains a UMAP reducer on the training set and transforms both train and test sets.
    Appends the UMAP components to the original features.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.

    Returns:
        tuple: Augmented X_train, Augmented X_test, and list of UMAP column names.
    """
    logger.info("Initializing and fitting UMAP...")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42, n_jobs=1)

    X_train_umap = umap_reducer.fit_transform(X_train)
    X_train_augmented = X_train.copy()
    X_train_augmented[umap_cols] = pd.DataFrame(X_train_umap, index=X_train.index)

    logger.info("Transforming test set with UMAP...")
    X_test_umap = umap_reducer.transform(X_test)
    X_test_augmented = X_test.copy()
    X_test_augmented[umap_cols] = pd.DataFrame(X_test_umap, index=X_test.index)

    return X_train_augmented, X_test_augmented, umap_cols