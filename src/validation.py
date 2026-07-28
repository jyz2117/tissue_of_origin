import logging
import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.metrics import accuracy_score

from src.config import models_to_skip, models, augmented_models

logger = logging.getLogger(__name__)

def test_pipeline(trained_models: dict, trained_augmented_models: dict, metaclassifier, X_test: pd.DataFrame, X_test_augmented: pd.DataFrame) -> np.ndarray:
    """
    Evaluates the pipeline on the test set, creating predictions from stage 1 models and passing them to stage 2.

    Args:
        trained_models (dict): Dictionary of trained standard base models.
        trained_augmented_models (dict): Dictionary of trained augmented base models.
        metaclassifier: Fitted stage 2 model.
        X_test (pd.DataFrame): Test feature matrix.
        X_test_augmented (pd.DataFrame): Test feature matrix with UMAP components.

    Returns:
        np.ndarray: Final class predictions from the metaclassifier.
    """
    logger.info("Running Stage 1 test predictions...")
    final_stage1_test_features = []

    for name in models.keys():
        if name.startswith(models_to_skip):
            continue
        model_fold_probs = []
        specific_trained_models = [m for k, m in trained_models.items() if k.startswith(name)]
        
        for model in specific_trained_models:
            probs = model.predict_proba(X_test)
            model_fold_probs.append(probs)
            
        stacked_probs = np.stack(model_fold_probs)
        aggregated_prob = gmean(stacked_probs, axis=0)
        
        df_model_probs = pd.DataFrame(
            aggregated_prob, 
            index=X_test.index,
            columns=[f"{name}_Class{c}" for c in range(aggregated_prob.shape[1])]
        )
        final_stage1_test_features.append(df_model_probs)

    for name in augmented_models.keys():
        if name.startswith(models_to_skip):
                continue
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

    logger.info("Generating Stage 2 test predictions...")
    X_test_stage2 = pd.concat(final_stage1_test_features, axis=1)
    stage2_preds = metaclassifier.predict(X_test_stage2)

    return stage2_preds

def validate_metaclassifier(y_test: pd.Series, stage2_preds: np.ndarray) -> None:
    """Logs the final accuracy score of the metaclassifier."""
    acc = accuracy_score(y_test, stage2_preds)
    logger.info(f"Metaclassifier Final Accuracy: {acc:.4f}") 

def validate_stage1(
    trained_models: dict, 
    trained_augmented_models: dict, 
    X_test: pd.DataFrame, 
    X_test_augmented: pd.DataFrame, 
    y_test: pd.Series
) -> None:
    """
    Logs the average Stage 1 test accuracy for each model family across all trained fold instances.
    """
    logger.info("Validating Stage 1 model families (average score across folds)...")

    # Evaluate standard base models
    for base_name in models.keys():
        if base_name.startswith(models_to_skip):
            continue

        scores = [
            model.score(X_test, y_test)
            for key, model in trained_models.items()
            if key.startswith(base_name)
        ]
        if scores:
            avg_score = np.mean(scores)
            logger.info(f"{base_name.ljust(20)} Mean Accuracy: {avg_score:.4f}")

    # Evaluate augmented base models (UMAP)
    for base_name in augmented_models.keys():
        if base_name.startswith(models_to_skip):
            continue

        scores = [
            model.score(X_test_augmented, y_test)
            for key, model in trained_augmented_models.items()
            if key.startswith(base_name)
        ]
        if scores:
            avg_score = np.mean(scores)
            logger.info(f"{base_name.ljust(20)} Mean Accuracy: {avg_score:.4f}")