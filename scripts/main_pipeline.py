import logging
import joblib
import time

from src.config import metaclassifier, seed, DATA_DIR, MODELS_DIR
from src.utils import time_convert
from src.data_prep import load_data, outer_split
from src.umap_aug import umap_aug
from src.stage1_train import stage1_train
from src.stage2_train import stage2_input, stage2_train
from src.validation import test_pipeline, validate_metaclassifier, validate_stage1

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Pipeline START")

    X, y = load_data('dat.csv', 'samples.csv')
    X_train, X_test, y_train, y_test = outer_split(X, y, test_size=0.2, random_state=seed)

    logger.info("Starting UMAP Augmentation...")
    start_time = time.perf_counter()
    X_train_augmented, X_test_augmented, umap_cols = umap_aug(X_train, X_test)
    logger.info(f"UMAP total time: {time_convert(time.perf_counter() - start_time)}")

    logger.info("Initiating STAGE 1 TRAINING...")
    sample_predictions, trained_models, trained_augmented_models = stage1_train(
        X_train, 
        X_train_augmented, 
        y_train, 
        umap_cols
    )
    
    # Save models and predictions using config paths
    joblib.dump((trained_models, trained_augmented_models), MODELS_DIR / "all_trained_models.joblib")
    joblib.dump(sample_predictions, DATA_DIR / "sample_predictions.joblib")

    logger.info("Initiating STAGE 2 TRAINING...")
    X_train_stage2 = stage2_input(sample_predictions, X_train, y_train)
    xgb_metaclassifier = stage2_train(X_train_stage2, y_train)

    joblib.dump(xgb_metaclassifier, MODELS_DIR / "metaclassifier.joblib")

    logger.info("Initiating Test Pipeline Validation...")
    stage2_preds = test_pipeline(
        trained_models, 
        trained_augmented_models, 
        metaclassifier, 
        X_test, 
        X_test_augmented
    )

    validate_metaclassifier(y_test, stage2_preds)
    validate_stage1(trained_models, trained_augmented_models, X_test, X_test_augmented, y_test)

    logger.info("Pipeline COMPLETE")
    return X_test, X_test_augmented, y_test, xgb_metaclassifier, trained_models, trained_augmented_models, sample_predictions

if __name__ == "__main__":
    main()