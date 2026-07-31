from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration Variables ---
seed = 42
umap_cols = ['A', "B", "C"] 
models_to_skip = () 

# --- Models ---
models = {
    "RandomForest": RandomForestClassifier(random_state=seed), 
    "ExtraTrees": ExtraTreesClassifier(random_state=seed),
    "GLM_Logistic": LogisticRegression(max_iter=5000, random_state=seed),
    "SVM_Linear": CalibratedClassifierCV(
        estimator=LinearSVC(
            C=50, 
            penalty='l1', 
            dual=False, 
            loss='squared_hinge',
            random_state=seed
        ),
        method='sigmoid', 
        cv=5
    )
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
    ),
    "MLP1": MLPClassifier(
        hidden_layer_sizes=(512, 512, 512),
        activation='relu',                   
        solver='adam',                       
        max_iter=1000,                        
        random_state=seed                      
    ),
    "MLP2": MLPClassifier(
        hidden_layer_sizes=(512, 512, 512, 512),
        activation='relu',                       
        solver='adam',                           
        max_iter=1000,                            
        random_state=seed                          
    )
}

metaclassifier = XGBClassifier(
    objective='multi:softmax', 
    learning_rate=0.1, 
    max_depth=5, 
    n_estimators=1000,
    subsample=0.9, 
    colsample_bytree=0.7,
    random_state=seed
)