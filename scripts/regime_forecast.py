"""
regime_forecast.py
==================
Dual regime forecasting: predict BOTH the economic regime (FE)
and the dynamic regime (MOMENT) at t+1 from features at t.

Uses a temporal train/test split (80/20) to avoid look-ahead bias.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = Path("results")
SEED = 42


def main():
    t0 = time.time()
    print("=" * 70)
    print("  DUAL REGIME FORECASTING")
    print("  Predict economic (FE) and dynamic (MOMENT) regime at t+1")
    print("=" * 70)

    # ── Load data ──
    lab_fe = pd.read_parquet(RESULTS / "exp_FE/step04/labels.parquet")
    lab_mom = pd.read_parquet(RESULTS / "exp_C/step04/labels.parquet")
    fe_emb = pd.read_parquet(RESULTS / "exp_FE/embeddings.parquet")

    for df in [lab_fe, lab_mom, fe_emb]:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Merge
    df = lab_fe.merge(lab_mom, on="datetime", suffixes=("_fe", "_mom"))
    df = df.merge(fe_emb, on="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Order regimes by mean LMP for interpretability
    pre = pd.read_parquet(RESULTS / "preprocessed.parquet")
    pre["datetime"] = pd.to_datetime(pre["datetime"])
    df = df.merge(pre[["datetime", "lmp"]], on="datetime")

    fe_order = df.groupby("cluster_fe")["lmp"].mean().sort_values().index.tolist()
    fe_rank = {r: i for i, r in enumerate(fe_order)}
    df["regime_econ"] = df["cluster_fe"].map(fe_rank)

    mom_order = df.groupby("cluster_mom")["lmp"].mean().sort_values().index.tolist()
    mom_rank = {r: i for i, r in enumerate(mom_order)}
    df["regime_dyn"] = df["cluster_mom"].map(mom_rank)

    K_econ = len(fe_order)
    K_dyn = len(mom_order)
    print(f"  Windows: {len(df)}")
    print(f"  Economic regimes (FE): {K_econ}")
    print(f"  Dynamic regimes (MOMENT): {K_dyn}")

    # ── Build features and targets ──
    # Features at time t: 19 FE features + current economic regime + current dynamic regime
    feature_cols = [c for c in fe_emb.columns if c != "datetime"]
    X_base = df[feature_cols].values

    # Add current regimes as features (one-hot)
    econ_onehot = pd.get_dummies(df["regime_econ"], prefix="econ").values
    dyn_onehot = pd.get_dummies(df["regime_dyn"], prefix="dyn").values

    X = np.hstack([X_base, econ_onehot, dyn_onehot])

    # Targets at t+1
    y_econ = df["regime_econ"].values[1:]  # next economic regime
    y_dyn = df["regime_dyn"].values[1:]    # next dynamic regime
    X = X[:-1]  # drop last (no target)

    n = len(X)
    print(f"  Feature dim: {X.shape[1]} (19 FE + {K_econ} econ OH + {K_dyn} dyn OH)")
    print(f"  Samples: {n}")

    # ── Temporal split 80/20 ──
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_econ_train, y_econ_test = y_econ[:split], y_econ[split:]
    y_dyn_train, y_dyn_test = y_dyn[:split], y_dyn[split:]

    print(f"  Train: {split} | Test: {n - split}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Baselines ──
    # Naive: predict same regime as current (persistence)
    y_econ_naive = df["regime_econ"].values[split:-1]  # current regime
    y_dyn_naive = df["regime_dyn"].values[split:-1]

    acc_econ_naive = accuracy_score(y_econ_test, y_econ_naive)
    acc_dyn_naive = accuracy_score(y_dyn_test, y_dyn_naive)
    f1_econ_naive = f1_score(y_econ_test, y_econ_naive, average="weighted")
    f1_dyn_naive = f1_score(y_dyn_test, y_dyn_naive, average="weighted")

    # Markov: predict most likely next regime from transition matrix
    def markov_predict(y_train, y_current, K):
        # Build transition matrix from training data
        T = np.zeros((K, K))
        for i in range(len(y_train) - 1):
            T[y_train[i], y_train[i + 1]] += 1
        T = T / T.sum(axis=1, keepdims=True).clip(1e-10)
        # Predict: most probable next state given current
        return np.array([np.argmax(T[c]) for c in y_current])

    y_econ_markov = markov_predict(y_econ_train, y_econ_naive, K_econ)
    y_dyn_markov = markov_predict(y_dyn_train, y_dyn_naive, K_dyn)

    acc_econ_markov = accuracy_score(y_econ_test, y_econ_markov)
    acc_dyn_markov = accuracy_score(y_dyn_test, y_dyn_markov)
    f1_econ_markov = f1_score(y_econ_test, y_econ_markov, average="weighted")
    f1_dyn_markov = f1_score(y_dyn_test, y_dyn_markov, average="weighted")

    # ── ML Models ──
    models = {
        "Logistic": LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                                random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                        random_state=SEED),
    }

    results = []

    # Add baselines
    results.append(("Naive (persistence)", acc_econ_naive, f1_econ_naive,
                     acc_dyn_naive, f1_dyn_naive))
    results.append(("Markov (1-step)", acc_econ_markov, f1_econ_markov,
                     acc_dyn_markov, f1_dyn_markov))

    for name, model in models.items():
        t1 = time.time()

        # Economic regime prediction
        m_econ = type(model)(**model.get_params())
        m_econ.fit(X_train_s, y_econ_train)
        y_econ_pred = m_econ.predict(X_test_s)
        acc_e = accuracy_score(y_econ_test, y_econ_pred)
        f1_e = f1_score(y_econ_test, y_econ_pred, average="weighted")

        # Dynamic regime prediction
        m_dyn = type(model)(**model.get_params())
        m_dyn.fit(X_train_s, y_dyn_train)
        y_dyn_pred = m_dyn.predict(X_test_s)
        acc_d = accuracy_score(y_dyn_test, y_dyn_pred)
        f1_d = f1_score(y_dyn_test, y_dyn_pred, average="weighted")

        elapsed = time.time() - t1
        results.append((name, acc_e, f1_e, acc_d, f1_d))
        print(f"  {name}: econ acc={acc_e:.3f} f1={f1_e:.3f} | "
              f"dyn acc={acc_d:.3f} f1={f1_d:.3f} ({elapsed:.1f}s)")

        # Feature importance for RF
        if name == "RandomForest":
            all_feat_names = list(feature_cols) + \
                [f"econ_{i}" for i in range(K_econ)] + \
                [f"dyn_{i}" for i in range(K_dyn)]

            print(f"\n  Top-10 features for ECONOMIC regime prediction:")
            imp_e = m_econ.feature_importances_
            top_e = np.argsort(imp_e)[::-1][:10]
            for rank, idx in enumerate(top_e):
                print(f"    {rank+1}. {all_feat_names[idx]:<20s} {imp_e[idx]:.4f}")

            print(f"\n  Top-10 features for DYNAMIC regime prediction:")
            imp_d = m_dyn.feature_importances_
            top_d = np.argsort(imp_d)[::-1][:10]
            for rank, idx in enumerate(top_d):
                print(f"    {rank+1}. {all_feat_names[idx]:<20s} {imp_d[idx]:.4f}")

    # ── Summary table ──
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Econ Acc':>8s} {'Econ F1':>8s} "
          f"{'Dyn Acc':>8s} {'Dyn F1':>8s}")
    print("  " + "-" * 60)
    for name, acc_e, f1_e, acc_d, f1_d in results:
        print(f"  {name:<25s} {acc_e:>8.3f} {f1_e:>8.3f} "
              f"{acc_d:>8.3f} {f1_d:>8.3f}")

    # ── Multi-step forecasting (Markov) ──
    print()
    print("-" * 70)
    print("  MULTI-STEP MARKOV FORECAST (how far ahead can we predict?)")
    print("-" * 70)

    for axis_name, y_full, K in [("Economic", df["regime_econ"].values, K_econ),
                                   ("Dynamic", df["regime_dyn"].values, K_dyn)]:
        # Build transition matrix from training data
        y_train_full = y_full[:split + 1]
        T = np.zeros((K, K))
        for i in range(len(y_train_full) - 1):
            T[y_train_full[i], y_train_full[i + 1]] += 1
        T = T / T.sum(axis=1, keepdims=True).clip(1e-10)

        print(f"\n  {axis_name} axis (K={K}):")
        for h in [1, 2, 4, 8, 16, 32]:
            T_h = np.linalg.matrix_power(T, h)
            # Predict h steps ahead
            y_current = y_full[split:n - h]
            y_target = y_full[split + h:n]
            if len(y_current) == 0:
                break
            y_pred = np.array([np.argmax(T_h[c]) for c in y_current])
            acc = accuracy_score(y_target, y_pred)
            # Stationary baseline
            y_naive = y_current  # persistence
            acc_naive = accuracy_score(y_target, y_naive)
            print(f"    h={h:>2d} steps ({h*6:>3d}h): "
                  f"Markov acc={acc:.3f}  Persistence acc={acc_naive:.3f}  "
                  f"Lift={acc - acc_naive:+.3f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
