import logging
import pandas as pd
from pathlib import Path

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import seed, DATA_DIR

logger = logging.getLogger(__name__)

def load_data(dat_file: str, samples_file: str, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads feature and target datasets from CSV files.

    Args:
        dat_file (str): Filename of the features CSV.
        samples_file (str): Filename of the target/samples CSV.
        data_dir (Path, optional): Directory containing the data. Defaults to DATA_DIR.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The feature matrix X and target dataframe y.
    """
    logger.info(f"Loading data from {data_dir}...")
    X = pd.read_csv(data_dir / dat_file, index_col=0)
    y = pd.read_csv(data_dir / samples_file, index_col=0)
    return X, y

def outer_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, random_state: int = seed) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a stratified train-test split based on age, sex, and disease state.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target dataframe containing patient metadata.
        test_size (float, optional): Proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed. Defaults to seed.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Performing outer train-test split...")
    y = y.copy()

    le = LabelEncoder()
    y['disease.state2'] = le.fit_transform(y['disease.state'])

    y['age_group'] = pd.cut(y['age'], bins=[0, 40, 50, 60, 70, 80, 100], labels=[1, 2, 3, 4, 5, 6])
    y['strata'] = y['age_group'].astype(str) + "_" + y['Sex'] + "_" + y['disease.state']

    counts = y['strata'].value_counts()
    y.loc[y['strata'].isin(counts[counts < 5].index), 'strata'] = 'rare'
    y['strata'] = y['strata'].fillna('rare')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y['strata'],
        random_state=random_state
    )

    y['set'] = "test"
    y.loc[y_train.index, 'set'] = "train"

    y_train_series = y_train['disease.state2']
    y_test_series = y_test['disease.state2']

    return X_train, X_test, y_train_series, y_test_series

def outer_split_summary(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Logs summary statistics comparing train and test distributions.
    """
    logger.info("Generating split summary statistics...")
    logger.info(f"Sex difference (Train - Test): {(y_train['Sex'] == 'Female').sum() / y_train.shape[0] - (y_test['Sex'] == 'Female').sum() / y_test.shape[0]}")
    logger.info(f"Row count difference (X_train vs y_train): {X_train.shape[0] - y_train.shape[0]}")
    logger.info(f"Row count difference (X_test vs y_test): {X_test.shape[0] - y_test.shape[0]}")