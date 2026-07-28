import logging
import joblib

from src.config import DATA_DIR, MODELS_DIR
from src.umap_aug import transform_umap_test
from src.validation import test_pipeline, validate_metaclassifier, validate_stage1

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("VALIDATION START")

    # 1. Load data
    logger.info("LOAD X_test, y_test, and all trained models")
    X_test = joblib.load(DATA_DIR / "X_test.joblib")
    y_test = joblib.load(DATA_DIR / "y_test.joblib")
    
    trained_models, trained_augmented_models = joblib.load(MODELS_DIR / "all_trained_models.joblib")
    metaclassifier = joblib.load(MODELS_DIR / "metaclassifier.joblib")

    # 2. UMAP aug (Inference Only)
    logger.info("UMAP transformation on X_test")
    X_test_augmented = transform_umap_test(X_test)

    # 3. validation
    logger.info("Predictions and stacking for stage 2")
    stage2_preds = test_pipeline(
        trained_models, 
        trained_augmented_models, 
        metaclassifier, 
        X_test, 
        X_test_augmented
    )

    # 4. accuracy scoring
    logger.info("Validate stage 2 and stage 1")
    validate_metaclassifier(y_test, stage2_preds)
    validate_stage1(trained_models, trained_augmented_models, X_test, X_test_augmented, y_test)

    logger.info("END")

if __name__ == "__main__":
    main()