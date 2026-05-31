"""Training loop, evaluation, and metrics for RUL prediction."""
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_rul: int = 30,
    include_phm: bool = False,
) -> dict:
    """Compute RMSE, MAE, and critical-zone versions. Optionally include PHM score.
    
    Critical zone = samples where y_true <= critical_rul.
    PHM score uses asymmetric exponential penalty:
        d = y_pred - y_true
        score = sum(exp(-d/13)-1 for d<0) + sum(exp(d/10)-1 for d>=0)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rmse = math.sqrt(np.mean((y_pred - y_true) ** 2))
    mae  = np.mean(np.abs(y_pred - y_true))

    mask = y_true <= critical_rul
    if mask.sum() > 0:
        rmse_crit = math.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))
        mae_crit  = np.mean(np.abs(y_pred[mask] - y_true[mask]))
    else:
        rmse_crit = float('nan')
        mae_crit  = float('nan')

    result = {'rmse': rmse, 'mae': mae, 'rmse_critical': rmse_crit, 'mae_critical': mae_crit}

    if include_phm:
        result['phm_score'] = _phm_score(y_true, y_pred)

    return result


def _phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(score.sum())


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    n_features: int = 24,
    n_layer: int = 2,
    num_heads: int = 4,
    head_dim: int = 16,
    dropout: float = 0.1,
    max_position_embeddings: int = 256,
) -> nn.Module:
    """Instantiate a model by name. model_name in {path, rope, alibi, nope, lstm, tcn}."""
    _dir = os.path.dirname(__file__)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from models import PaTHRUL, TransformerRUL, LSTM_RUL, TCN_RUL

    hidden_size = num_heads * head_dim
    base_kw = dict(n_features=n_features, n_layer=n_layer, num_heads=num_heads,
                   head_dim=head_dim, dropout=dropout,
                   max_position_embeddings=max_position_embeddings)
    if model_name == 'path':
        return PaTHRUL(**base_kw)
    if model_name == 'rope':
        return TransformerRUL(pe_method='rotary', **base_kw)
    if model_name == 'alibi':
        return TransformerRUL(pe_method='alibi', **base_kw)
    if model_name == 'nope':
        return TransformerRUL(pe_method='no_pe', **base_kw)
    if model_name == 'lstm':
        return LSTM_RUL(n_features=n_features, hidden_size=hidden_size,
                        num_layers=n_layer, dropout=dropout)
    if model_name == 'tcn':
        return TCN_RUL(n_features=n_features,
                       channels=(hidden_size,) * 3, dropout=dropout)
    raise ValueError(f'Unknown model_name: {model_name}')


def _collate_fn(batch):
    """Collate (x, y, mask_or_None) batches. Stacks masks when present."""
    xs, ys, masks = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    if masks[0] is not None:
        mask = torch.stack(masks)
    else:
        mask = None
    return x, y, mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_rul: int = 125,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for x, y, mask in loader:
        x, y = x.to(device), y.to(device)
        mask = mask.to(device) if mask is not None else None
        optimizer.zero_grad()
        pred = model(x, attention_mask=mask)
        loss = criterion(pred, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_rul: int = 125,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference. Returns (y_true, y_pred) clipped to [0, max_rul]."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            mask = mask.to(device) if mask is not None else None
            pred = model(x, attention_mask=mask).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.clip(np.concatenate(all_pred), 0.0, max_rul)
    return y_true, y_pred


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    critical_rul: int = 30,
    include_phm: bool = False,
    max_rul: int = 125,
) -> dict:
    y_true, y_pred = predict(model, loader, device, max_rul)
    return compute_metrics(y_true, y_pred, critical_rul=critical_rul, include_phm=include_phm)


def run_experiment(args) -> dict:
    """Full pipeline: data → model → training → evaluation → results.json."""
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # data
    sys.path.insert(0, os.path.dirname(__file__))
    from cmapss_dataset import prepare_fd001_datasets
    train_ds, val_ds, test_ds, meta = prepare_fd001_datasets(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_rul=args.max_rul,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     collate_fn=_collate_fn, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # model
    n_features = len(meta['feature_cols'])
    model = build_model(
        args.model, n_features=n_features, n_layer=args.n_layer,
        num_heads=args.num_heads, head_dim=args.head_dim,
        dropout=args.dropout, max_position_embeddings=args.max_position_embeddings,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_rmse = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.max_rul, args.grad_clip)
        val_metrics = evaluate(model, val_loader, device,
                               critical_rul=args.critical_rul,
                               include_phm=False, max_rul=args.max_rul)
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f'epoch {epoch:3d}  train_loss={train_loss:.4f}  '
                  f'val_rmse={val_metrics["rmse"]:.4f}')

    # load best checkpoint for test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device,
                            critical_rul=args.critical_rul,
                            include_phm=args.include_phm, max_rul=args.max_rul)

    results = {
        'model': args.model,
        'best_val_rmse': best_val_rmse,
        'test': test_metrics,
        'args': vars(args),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'results_{args.model}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {out_path}')
    print('Test metrics:', test_metrics)
    return results
