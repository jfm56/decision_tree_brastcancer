import pandas as pd

BIN_LABELS = ['low', 'high']

# Feature binning for all columns except target
def bin_features(dataframe, feature_columns):
    """Bin features into quantiles labeled as low, medium, high."""
    binned_dataframe = dataframe.copy()
    for column in feature_columns:
        binned_dataframe[column] = pd.qcut(
            binned_dataframe[column], q=2, labels=BIN_LABELS
        )
    return binned_dataframe

def map_target(dataframe, target_column='diagnosis'):
    # Map target values to more descriptive labels
    return dataframe[target_column].map({'M': 'malignant', 'B': 'benign'})

def preprocess_data(df, feature_columns, target_column='diagnosis'):
    # Preprocess data for tree: use continuous features, only map target values
    df_processed = df.copy()
    df_processed[target_column] = map_target(df_processed, target_column)
    return df_processed
