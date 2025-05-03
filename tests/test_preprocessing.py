import pandas as pd
from src import preprocessing

def test_bin_features():
    dataframe = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6],
        'b': [10, 20, 30, 40, 50, 60]
    })
    binned = preprocessing.bin_features(dataframe, ['a', 'b'])
    assert set(binned['a'].unique()) <= set(preprocessing.BIN_LABELS)
    assert set(binned['b'].unique()) <= set(preprocessing.BIN_LABELS)

def test_map_target():
    dataframe = pd.DataFrame({'diagnosis': ['M', 'B', 'M']})
    mapped = preprocessing.map_target(dataframe)
    assert list(mapped) == ['malignant', 'benign', 'malignant']
