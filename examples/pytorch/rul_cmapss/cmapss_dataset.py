import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
    """Compute mean and std over training rows. Near-zero std → 1.0."""
    vals = train_df[feature_cols].values.astype(np.float32)
    mean = vals.mean(axis=0).astype(np.float32)
    std  = vals.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def log_feature_stats(mean: np.ndarray, std: np.ndarray, train_df: pd.DataFrame,
                      feature_cols: list) -> None:
    """Print per-feature statistics for debugging."""
    vals = train_df[feature_cols].values.astype(np.float32)
    print(f"{'feature':<22}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}  {'n_unique':>8}")
    for i, col in enumerate(feature_cols):
        n_uniq = int(np.unique(vals[:, i]).size)
        print(f"{col:<22}  {mean[i]:>9.4f}  {std[i]:>9.4f}  {vals[:,i].min():>9.4f}"
              f"  {vals[:,i].max():>9.4f}  {n_uniq:>8d}")


def apply_standardizer(df: pd.DataFrame, feature_cols: list, mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    """Z-score normalize feature columns in-place copy."""
    df = df.copy()
    df[feature_cols] = (df[feature_cols].values.astype(np.float32) - mean) / std
    return df


class WindowedRULDataset(Dataset):
    """Sliding-window RUL dataset. mask is None for fully-observed windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32) if mask is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        mask = self.mask[idx] if self.mask is not None else None
        return self.x[idx], self.y[idx], mask


def build_train_windows(
    train_df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide window over each engine's sequence.
    
    Returns x[N, T, F], y[N] where y is RUL at the last timestep of each window.
    train_df must already have a 'rul' column.
    """
    xs, ys = [], []
    for _, seq in train_df.groupby('engine_id'):
        seq = seq.sort_values('cycle')
        feats = seq[feature_cols].values.astype(np.float32)
        ruls  = seq['rul'].values.astype(np.float32)
        n = len(feats)
        for end in range(window_size - 1, n, stride):
            xs.append(feats[end - window_size + 1 : end + 1])
            ys.append(ruls[end])
    if not xs:
        raise ValueError(
            f"No windows produced: all engines shorter than window_size={window_size}. "
            "Try reducing --window_size."
        )
    return np.stack(xs), np.array(ys, dtype=np.float32)


def build_test_windows(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
    max_rul: int = 125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one window per test engine. Right-zero-pads sequences shorter than window_size.

    Returns engine_ids[K], x[K,T,F], y[K], masks[K,T].
    masks[i,t] = 1 for real timestep, 0 for right-pad.

    Right-padding is used so that under causal attention the last real timestep
    (position n-1) never attends to padded positions — safe for PaTH which
    ignores attention_mask. The regression head uses the last *real* position
    (masks.sum()-1), not always position T-1.
    """
    engine_ids, xs, ys, masks = [], [], [], []
    rul_lookup = dict(zip(rul_df['engine_id'], rul_df['rul']))
    for eid, seq in test_df.groupby('engine_id'):
        seq = seq.sort_values('cycle')
        feats = seq[feature_cols].values.astype(np.float32)
        n = len(feats)
        if n >= window_size:
            win  = feats[-window_size:]
            mask = np.ones(window_size, dtype=np.float32)
        else:
            pad  = np.zeros((window_size - n, len(feature_cols)), dtype=np.float32)
            win  = np.concatenate([feats, pad], axis=0)   # real first, pad at end
            mask = np.concatenate([np.ones(n, dtype=np.float32),
                                   np.zeros(window_size - n, dtype=np.float32)])
        engine_ids.append(eid)
        xs.append(win)
        rul = rul_lookup.get(eid)
        if rul is None:
            raise KeyError(
                f"Engine {eid} in test file not found in RUL file. "
                f"RUL file has {len(rul_lookup)} engines: {sorted(rul_lookup)[:5]}..."
            )
        ys.append(min(float(rul), float(max_rul)))
        masks.append(mask)
    return (
        np.array(engine_ids),
        np.stack(xs),
        np.array(ys, dtype=np.float32),
        np.stack(masks),
    )


def prepare_fd001_datasets(
    data_dir: str,
    window_size: int = 30,
    stride: int = 1,
    max_rul: int = 125,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> tuple["WindowedRULDataset", "WindowedRULDataset", "WindowedRULDataset", dict]:
    """Full pipeline: load → label → split → normalize → build datasets.
    
    Engine-level train/val split prevents data leakage.
    Scaler is fit on training-engine rows only.
    """
    train_raw, test_raw, rul_df = load_fd001_raw(data_dir)
    train_labeled = add_piecewise_rul(train_raw, max_rul=max_rul)

    # engine-level split
    rng = np.random.default_rng(split_seed)
    engine_ids = train_labeled['engine_id'].unique()
    rng.shuffle(engine_ids)
    n_val = max(1, int(len(engine_ids) * val_ratio))
    val_engines   = set(engine_ids[:n_val])
    train_engines = set(engine_ids[n_val:])

    train_split = train_labeled[train_labeled['engine_id'].isin(train_engines)]
    val_split   = train_labeled[train_labeled['engine_id'].isin(val_engines)]

    mean, std = fit_standardizer(train_split, FEATURE_COLS)
    train_norm = apply_standardizer(train_split, FEATURE_COLS, mean, std)
    val_norm   = apply_standardizer(val_split,   FEATURE_COLS, mean, std)
    test_norm  = apply_standardizer(test_raw,    FEATURE_COLS, mean, std)

    x_tr, y_tr = build_train_windows(train_norm, FEATURE_COLS, window_size, stride)
    x_va, y_va = build_train_windows(val_norm,   FEATURE_COLS, window_size, stride)
    _, x_te, y_te, te_masks = build_test_windows(test_norm, rul_df, FEATURE_COLS, window_size, max_rul)

    train_ds = WindowedRULDataset(x_tr, y_tr)
    val_ds   = WindowedRULDataset(x_va, y_va)
    test_ds  = WindowedRULDataset(x_te, y_te, te_masks)

    meta = {
        'feature_cols': FEATURE_COLS,
        'mean': mean,
        'std': std,
        'train_engines': [int(x) for x in sorted(train_engines)],
        'val_engines': [int(x) for x in sorted(val_engines)],
    }
    return train_ds, val_ds, test_ds, meta
