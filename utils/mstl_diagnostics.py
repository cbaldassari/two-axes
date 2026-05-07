"""
mstl_diagnostics.py
===================
MSTL deseasonalization diagnostics.

Before applying MSTL to a channel, check whether detrending is appropriate:
- If residual variance is >80% of total -> MSTL is not capturing meaningful seasonality
- If seasonal ACF at lag=period is weak (<0.10) -> no real seasonal pattern

Channels that fail these tests should keep raw values (skip detrending).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL


def diagnose_channel(series: np.ndarray,
                     periods: list[int],
                     windows: list[int]) -> dict:
    """
    Run MSTL on a single channel and compute diagnostics.

    Parameters
    ----------
    series : 1D array (hourly, no NaN)
    periods : MSTL seasonal periods (e.g., [24, 168])
    windows : MSTL window sizes (e.g., [25, 169])

    Returns
    -------
    dict with:
      residual_var_ratio : float — var(residual) / var(series)
      seasonal_acf : dict — {period: acf_at_lag_period}
      total_var : float
      residual_var : float
    """
    s = pd.Series(series.astype(np.float64))
    total_var = float(s.var())

    if total_var < 1e-12:
        return {
            "residual_var_ratio": 1.0,
            "seasonal_acf": {p: 0.0 for p in periods},
            "total_var": total_var,
            "residual_var": total_var,
        }

    # Fit MSTL
    mstl = MSTL(s, periods=periods, windows=windows)
    result = mstl.fit()

    residual = result.resid.values
    residual_var = float(np.nanvar(residual))
    residual_var_ratio = residual_var / total_var

    # Seasonal ACF at lag = period
    seasonal_acf = {}
    for p in periods:
        seasonal_comp = None
        for i, sp in enumerate(result.seasonal.columns if hasattr(result.seasonal, 'columns') else range(len(periods))):
            if i < len(periods) and periods[i] == p:
                if hasattr(result.seasonal, 'iloc'):
                    seasonal_comp = result.seasonal.iloc[:, i].values
                else:
                    seasonal_comp = result.seasonal[:, i]
                break

        if seasonal_comp is not None and len(seasonal_comp) > p:
            # ACF at lag = period
            n = len(seasonal_comp)
            mean = np.nanmean(seasonal_comp)
            var = np.nanvar(seasonal_comp)
            if var > 1e-12:
                cov = np.nanmean((seasonal_comp[p:] - mean) * (seasonal_comp[:-p] - mean))
                acf = cov / var
            else:
                acf = 0.0
            seasonal_acf[p] = float(acf)
        else:
            seasonal_acf[p] = 0.0

    return {
        "residual_var_ratio": round(residual_var_ratio, 4),
        "seasonal_acf": seasonal_acf,
        "total_var": round(total_var, 6),
        "residual_var": round(residual_var, 6),
    }


def run_all_diagnostics(df: pd.DataFrame,
                        channels: list[str],
                        periods: list[int],
                        windows: list[int],
                        max_residual_var_ratio: float = 0.80,
                        min_seasonal_acf: float = 0.10) -> dict[str, dict]:
    """
    Run MSTL diagnostics on all channels.

    Parameters
    ----------
    df : DataFrame with hourly data (columns must include all channels)
    channels : list of column names to diagnose
    periods, windows : MSTL parameters
    max_residual_var_ratio : skip detrending if ratio exceeds this
    min_seasonal_acf : skip detrending if ACF below this for ALL periods

    Returns
    -------
    dict[channel_name -> {diagnostics + skip_detrending: bool}]
    """
    results = {}
    for ch in channels:
        if ch not in df.columns:
            print(f"    [MSTL diag] Channel '{ch}' not in DataFrame, skipping", flush=True)
            continue

        series = df[ch].values.copy()

        # Fill NaN with median for diagnostics
        nan_mask = np.isnan(series)
        if nan_mask.any():
            series[nan_mask] = np.nanmedian(series)

        diag = diagnose_channel(series, periods, windows)

        # Decision: skip detrending?
        high_residual = diag["residual_var_ratio"] > max_residual_var_ratio
        weak_acf = all(v < min_seasonal_acf for v in diag["seasonal_acf"].values())
        skip = high_residual and weak_acf

        diag["skip_detrending"] = skip
        results[ch] = diag

        status = "SKIP" if skip else "DETREND"
        print(f"    [{status}] {ch}: resid_var={diag['residual_var_ratio']:.2%}  "
              f"acf={diag['seasonal_acf']}", flush=True)

    return results
