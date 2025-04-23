import pandas as pd
import numpy as np

BIN_LABELS = ['low', 'medium', 'high']

# Feature binning for all columns except target
def bin_features(df, feature_cols):
    binned_df = df.copy()
    for col in feature_cols:
        binned_df[col] = pd.qcut(binned_df[col], q=3, labels=BIN_LABELS)
    return binned_df

def map_target(df, target_col='diagnosis'):
    return df[target_col].map({'M': 'malignant', 'B': 'benign'})

def preprocess_data(df, feature_cols, target_col='diagnosis'):
    df_binned = bin_features(df, feature_cols)
    df_binned[target_col] = map_target(df_binned, target_col)
    return df_binned
