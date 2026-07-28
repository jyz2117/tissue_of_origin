import logging
import joblib
import time

from src.config import seed, DATA_DIR, MODELS_DIR
from src.utils import time_convert
from src.data_prep import load_data, outer_split
from src.umap_aug import fit_umap_train
from src.stage1_train import stage1_train
from src.stage2_train import stage2_input, stage2_train

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("TRAIN START")

    # 1. Load and Split Data
    X, y = load_data('dat.csv', 'samples.csv')
    X_train, X_test, y_train, y_test = outer_split(X, y, test_size=0.2, random_state=seed)

    # Save the test set for the validation script to pick up later
    joblib.dump(X_test, DATA_DIR / "X_test.joblib")
    joblib.dump(y_test, DATA_DIR / "y_test.joblib")
    logger.info("Saved X_test and y_test for validation.")

    # 2. UMAP Augmentation
    logger.info("UMAP fit and transform on X_train")
    start_time = time.perf_counter()
    X_train_augmented, umap_cols = fit_umap_train(X_train)
    logger.info(f"UMAP total time: {time_convert(time.perf_counter() - start_time)}")
    logger.info("Saved X_train_augmented")

    # 3. Stage 1 Training
    logger.info("STAGE 1 TRAINING and STACKING")
    sample_predictions, trained_models, trained_augmented_models = stage1_train(
        X_train, 
        X_train_augmented, 
        y_train, 
        umap_cols
    )
    
    joblib.dump((trained_models, trained_augmented_models), MODELS_DIR / "all_trained_models.joblib")
    joblib.dump(sample_predictions, DATA_DIR / "sample_predictions.joblib")
    logger.info("Saved all models and sample_predictions")

    # 4. Stage 2 Training
    logger.info("STAGE 2 XGB TRAINING")
    X_train_stage2 = stage2_input(sample_predictions, X_train, y_train)
    xgb_metaclassifier = stage2_train(X_train_stage2, y_train)

    joblib.dump(xgb_metaclassifier, MODELS_DIR / "metaclassifier.joblib")

    logger.info("END")

if __name__ == "__main__":
    main()