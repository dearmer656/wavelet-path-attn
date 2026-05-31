"""Training loop, evaluation, and metrics for RUL prediction."""
import math
import numpy as np


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
