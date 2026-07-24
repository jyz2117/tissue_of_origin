import pandas as pd
import numpy as np

from sklearnex import patch_sklearn # for performance
patch_sklearn()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import joblib

seed = 42

def time_convert(time):
    if time < 60:
        return f"{time:.2f} seconds"
    elif time < 3600:
        return f"{time/60:.2f} minutes"
    else:
        return f"{time/3600:.2f} hours"

            #################
            ### READ DATA ###
            #################

# TRAIN TEST SPLIT
X = pd.read_csv('dat.csv', index_col = 0) # convert first column to rownames automatically
y = pd.read_csv('samples.csv', index_col = 0)

le = LabelEncoder()
y['disease.state2'] = le.fit_transform(y['disease.state'])

# check for NA
print(X.isna().sum().value_counts())

y['age_group'] = pd.cut(y['age'], bins = [0, 40, 50, 60, 70, 80, 100], labels = [1, 2, 3, 4, 5, 6])
y['strata'] = y['age_group'].astype(str) + "_" + y['Sex'] + "_" + y['disease.state']

# handle rare cases in combo strata
counts = y['strata'].value_counts() # count of each strata
y.loc[y['strata'].isin(counts[counts < 5].index), 'strata'] = 'rare' # assign 'rare' to all straata where its count < 5
y['strata'] = y['strata'].fillna('rare') # assign 3 NA to 'rare'


            ########################
            ### TRAIN TEST SPLIT ###
            ########################

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y['strata'],
        random_state=seed
    )

# check proportions and length
print((y_train['Sex'] == 'Female').sum() / y_train.shape[0] - (y_test['Sex'] == 'Female').sum() / y_test.shape[0])
print(y_train['age_group'].value_counts() / y_train.shape[0] - y_test['age_group'].value_counts() / y_test.shape[0])
print(y_train['disease.state'].value_counts() / y_train.shape[0] - y_test['disease.state'].value_counts() / y_test.shape[0])
print(X_train.shape[0] - y_train.shape[0])
print(X_test.shape[0] - y_test.shape[0])

# create corresponding set column in samples
y['set'] = "test"
y.loc[y_train.index, 'set'] = "train"

y_train = y_train['disease.state2'] # isolate targets
y_test = y_test['disease.state2']

joblib.dump((X_train, X_test, y_train, y_test), "train_test_split.joblib")


