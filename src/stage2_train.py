import logging
import pandas as pd
import numpy as np
import joblib
from scipy.stats import gmean

from src.config import models_to_skip, models, augmented_models, metaclassifier, DATA_DIR

logger = logging.getLogger(__name__)

def stage2_input(sample_predictions: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Aggregates stage 1 model predictions using a geometric mean to create stage 2 inputs.

    Args:
        sample_predictions (dict): Dictionary of stage 1 probability arrays per sample.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.

    Returns:
        pd.DataFrame: Engineered features for the metaclassifier.
    """
    logger.info("Aggregating Stage 1 probabilities via geometric mean...")
    all_model_names = list(models.keys()) + list(augmented_models.keys())
    final_stage1_features = []

    for name in all_model_names:
        if name.startswith(models_to_skip):
            continue
        model_gmean_probs = []
        
        for sample_idx in X_train.index:
            four_probs = np.array(sample_predictions[name][sample_idx])
            aggregated_prob = gmean(four_probs, axis=0)
            model_gmean_probs.append(aggregated_prob)
            
        df_model_probs = pd.DataFrame(
            model_gmean_probs, 
            index=X_train.index,
            columns=[f"{name}_Class{c}" for c in range(len(model_gmean_probs[0]))]
        )
        final_stage1_features.append(df_model_probs)

    X_train_stage2 = pd.concat(final_stage1_features, axis=1)
    
    # Dump generated data
    joblib.dump(X_train_stage2, DATA_DIR / "X_train_stage2.joblib")

    return X_train_stage2

def stage2_train(X_train_stage2: pd.DataFrame, y_train: pd.Series):
    """
    Trains the XGBoost metaclassifier on the aggregated predictions.

    Args:
        X_train_stage2 (pd.DataFrame): Aggregated stage 1 predictions.
        y_train (pd.Series): Target labels.

    Returns:
        XGBClassifier: The fitted metaclassifier.
    """
    logger.info("Training Stage 2 metaclassifier...")
    metaclassifier.fit(X_train_stage2, y_train)
    return metaclassifier