"""
config.py
=========
Central configuration for the NEPOOL TDA Pipeline.
ISONE Electricity Market — Topological Regime Detection
"""

DATA_PATH    = "isone_dataset.parquet"
RESULTS_DIR  = "results"
RANDOM_STATE = 42

DATASET = {
    "start_date": "2021-01-01",
    "end_date":   "2025-12-31",
}

ILR = {
    "zero_replacement_delta": 0.0001,
    "fuel_cols": [
        "natural_gas_share",
        "nuclear_share",
        "hydro_share",
        "wind_share",
        "solar_share",
        "coal_share",
        "oil_share",
        "other_share",
    ],
}

EMBEDDING = {
    "model":       "AutonLab/MOMENT-1-large",
    "context_len": 512,
    "stride_h":    6,
    "batch_size":  64,
}

# 19 hand-crafted features computed per window
FE_FEATURES = [
    "mean", "std", "skew", "kurt", "min", "max", "range",
    "median", "p5", "p95", "iqr",
    "acf_1h", "acf_6h", "acf_24h", "acf_168h",
    "vol_24h",
    "lmp_mean", "lmp_p95", "lmp_std",
]

# 15 features for delta-arcsinh pipeline (no ACF)
FE_FEATURES_15 = [
    "mean", "std", "skew", "kurt", "min", "max", "range",
    "median", "p5", "p95", "iqr",
    "vol_24h",
    "lmp_mean", "lmp_p95", "lmp_std",
]

DARCSINH = {
    "mstl_periods": [24, 168, 8760],
    "W": 512,
    "S": 6,
}
