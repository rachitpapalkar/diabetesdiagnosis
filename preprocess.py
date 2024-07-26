import pandas as pd
import numpy as np

X = synthetic_data.drop(columns=['Outcome']).values
y = synthetic_data['Outcome'].values

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X_train = X[indices[:-n_test]]
    X_test = X[indices[-n_test:]]
    y_train = y[indices[:-n_test]]
    y_test = y[indices[-n_test:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

def min_max_scaling(X_train, X_test):
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    X_train = (X_train - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)
    return X_train, X_test

X_train, X_test = min_max_scaling(X_train, X_test)

df_train = pd.DataFrame(X_train, columns=synthetic_data.columns[:-1])
df_test = pd.DataFrame(X_test, columns=synthetic_data.columns[:-1])
df_train.to_csv('standardized_training.csv', index=False)
df_test.to_csv('standardized_testing.csv', index=False)
