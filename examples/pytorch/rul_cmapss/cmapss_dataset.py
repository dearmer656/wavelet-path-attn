import os

import numpy as np
import pandas as pd


FEATURE_COLS = ['op_setting_1','op_setting_2','op_setting_3'] + ['sensor_{}'.format(i) for i in range(1,22)]
# 3 op settings + 21 sensors = 24 features total


def load_fd001_raw(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw FD001 train, test, and RUL files.

    Files are whitespace-delimited, no header.
    Columns: engine_id, cycle, op_setting_1..3, sensor_1..21 (26 total)
    Returns: (train_df, test_df, rul_df) where rul_df has one row per engine
    """
    col_names = ['engine_id', 'cycle'] + FEATURE_COLS
    train_df = pd.read_csv(os.path.join(data_dir, 'train_FD001.txt'), sep=r'\s+', header=None, names=col_names)
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_FD001.txt'),  sep=r'\s+', header=None, names=col_names)
    rul_df   = pd.read_csv(os.path.join(data_dir, 'RUL_FD001.txt'),   sep=r'\s+', header=None, names=['rul'])
    rul_df['engine_id'] = rul_df.index + 1
    return train_df, test_df, rul_df


def add_piecewise_rul(train_df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Add 'rul' column to train_df using piecewise linear cap."""
    # per engine: rul = min(max_cycle - cycle, max_rul)
    max_cycles = train_df.groupby('engine_id')['cycle'].max().rename('max_cycle')
    df = train_df.merge(max_cycles, on='engine_id')
    df['rul'] = (df['max_cycle'] - df['cycle']).clip(upper=max_rul)
    return df.drop(columns=['max_cycle'])


def fit_standardizer(train_df: pd.DataFrame, feature_cols: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std over training rows. Zero std → 1.0."""
    vals = train_df[feature_cols].values.astype(np.float32)
    mean = vals.mean(axis=0)
    std  = vals.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def apply_standardizer(df: pd.DataFrame, feature_cols: list, mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    """Z-score normalize feature columns in-place copy."""
    df = df.copy()
    df[feature_cols] = (df[feature_cols].values.astype(np.float32) - mean) / std
    return df
