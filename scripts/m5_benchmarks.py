from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# ---------------------------------------------------------------------
# Optional metric imports.
# Assumes compute_m5_metrics.py is on your PYTHONPATH or in the same folder.
# ---------------------------------------------------------------------
try:
    from compute_m5_metrics import (
        compute_wrmsse_from_samples,
        compute_wspl_from_samples,
    )
except Exception:
    compute_wrmsse_from_samples = None
    compute_wspl_from_samples = None


# ============================================================
# 1) Basic helpers
# ============================================================

def _safe_std(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.std(x))


def _last_nonzero_count(y: np.ndarray, window: int) -> float:
    w = y[-window:] if y.shape[0] >= window else y
    return float(np.count_nonzero(w))


def _rolling_stats(y: np.ndarray, window: int) -> Tuple[float, float]:
    w = y[-window:] if y.shape[0] >= window else y
    return float(np.mean(w)), _safe_std(w)


def _topk_share(child_recent_sales: np.ndarray, k: int) -> float:
    total = float(child_recent_sales.sum())
    if total <= 0:
        return 0.0
    vals = np.sort(child_recent_sales)[::-1]
    return float(vals[: min(k, len(vals))].sum() / total)


def _hhi(shares: np.ndarray) -> float:
    s = shares.astype(float)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(np.sum(p ** 2))


def _seasonal_repeat(y_hist: np.ndarray, horizon: int, seasonality: int = 7) -> np.ndarray:
    """
    Weekly seasonal naive repeat.
    y_hist: [T]
    returns: [H]
    """
    if y_hist.size == 0:
        return np.zeros(horizon, dtype=np.float32)

    s = min(seasonality, y_hist.shape[0])
    pattern = y_hist[-s:]
    out = np.empty(horizon, dtype=np.float32)
    for h in range(horizon):
        out[h] = pattern[h % s]
    return out


def _future_mean_features(sample: Dict[str, Any]) -> np.ndarray:
    x_future = sample["x_future"]  # [N, H, F]
    return x_future.mean(axis=0).reshape(-1).astype(np.float32)


def _future_rich_summary_features(sample: Dict[str, Any]) -> np.ndarray:
    """
    Rich child-summary features over future-known covariates.
    For each horizon step and feature, use mean/std/q25/q50/q75/min/max.
    """
    x_future = sample["x_future"]  # [N, H, F]
    N, H, F = x_future.shape
    feats = []
    for h in range(H):
        step = x_future[:, h, :]  # [N, F]
        feats.append(step.mean(axis=0))
        feats.append(step.std(axis=0))
        feats.append(np.quantile(step, 0.25, axis=0))
        feats.append(np.quantile(step, 0.50, axis=0))
        feats.append(np.quantile(step, 0.75, axis=0))
        feats.append(step.min(axis=0))
        feats.append(step.max(axis=0))
    return np.concatenate(feats, axis=0).astype(np.float32)


def _aggregate_history_features(sample: Dict[str, Any]) -> np.ndarray:
    """
    Features from aggregate history only.
    """
    y_hist = sample["y_hist"].sum(axis=0).reshape(-1)  # [T]

    mean7, std7 = _rolling_stats(y_hist, 7)
    mean14, std14 = _rolling_stats(y_hist, 14)
    mean28, std28 = _rolling_stats(y_hist, 28)
    mean56, std56 = _rolling_stats(y_hist, 56)

    lag1 = float(y_hist[-1]) if y_hist.shape[0] >= 1 else 0.0
    lag7 = float(y_hist[-7]) if y_hist.shape[0] >= 7 else lag1
    lag14 = float(y_hist[-14]) if y_hist.shape[0] >= 14 else lag7
    lag28 = float(y_hist[-28]) if y_hist.shape[0] >= 28 else lag14

    feats = np.array([
        lag1, lag7, lag14, lag28,
        mean7, std7,
        mean14, std14,
        mean28, std28,
        mean56, std56,
        _last_nonzero_count(y_hist, 28),
        float(y_hist.sum()),
    ], dtype=np.float32)
    return feats


def _child_structure_features(sample: Dict[str, Any]) -> np.ndarray:
    """
    Structural features from child set.
    """
    y_hist_children = sample["y_hist"][:, :, 0]  # [N, T]
    N = y_hist_children.shape[0]

    recent_sales = y_hist_children[:, -28:].sum(axis=1) if y_hist_children.shape[1] >= 28 else y_hist_children.sum(axis=1)
    feats = np.array([
        float(N),
        float(recent_sales.mean()) if N > 0 else 0.0,
        float(recent_sales.std()) if N > 1 else 0.0,
        _topk_share(recent_sales, 1),
        _topk_share(recent_sales, 5),
        _topk_share(recent_sales, 10),
        _hhi(recent_sales),
    ], dtype=np.float32)
    return feats


# ============================================================
# 2) Feature builders
# ============================================================

def build_direct_aggregate_matrix(samples: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct aggregate baseline:
      - aggregate y history features
      - mean future-known features across children
    """
    X = []
    Y = []
    for s in samples:
        x = np.concatenate([
            _aggregate_history_features(s),
            _future_mean_features(s),
        ], axis=0)
        X.append(x)
        Y.append(s["y_target"].astype(np.float32))
    return np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32)


def build_child_summary_matrix(samples: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handcrafted cross-level benchmark:
      - aggregate y history features
      - rich future child summaries
      - child structure summaries
    """
    X = []
    Y = []
    for s in samples:
        x = np.concatenate([
            _aggregate_history_features(s),
            _future_rich_summary_features(s),
            _child_structure_features(s),
        ], axis=0)
        X.append(x)
        Y.append(s["y_target"].astype(np.float32))
    return np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32)


def _row_lag(y_hist: np.ndarray, lag: int) -> np.ndarray:
    """Vectorized lag feature for child-level demand histories."""
    if y_hist.shape[1] == 0:
        return np.zeros((y_hist.shape[0], 1), dtype=np.float32)
    idx = -lag if y_hist.shape[1] >= lag else -1
    return y_hist[:, idx:idx + 1].astype(np.float32, copy=False)


def _row_rolling_mean_std(y_hist: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized rolling mean/std features for child-level histories."""
    if y_hist.shape[1] == 0:
        z = np.zeros((y_hist.shape[0], 1), dtype=np.float32)
        return z, z
    w = y_hist[:, -window:] if y_hist.shape[1] >= window else y_hist
    return w.mean(axis=1, keepdims=True).astype(np.float32), w.std(axis=1, keepdims=True).astype(np.float32)


def build_bottom_level_global_features(samples: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
    """
    Child/store-SKU design matrix for the global bottom-up forecasting benchmark.

    Each row is one child store-SKU at one forecast origin. The fitted model predicts
    the 28-step child demand vector; predictions are summed back to the aggregate unit.
    """
    X_parts: List[np.ndarray] = []
    child_counts: List[int] = []

    for s in samples:
        y_hist = np.asarray(s["y_hist"][:, :, 0], dtype=np.float32)  # [N, T]
        x_hist = np.asarray(s.get("x_hist", np.zeros((y_hist.shape[0], y_hist.shape[1], 0), dtype=np.float32)), dtype=np.float32)
        x_future = np.asarray(s.get("x_future", np.zeros((y_hist.shape[0], s["y_target"].shape[0], 0), dtype=np.float32)), dtype=np.float32)
        N = int(y_hist.shape[0])
        child_counts.append(N)

        demand_feats = []
        for lag in (1, 7, 14, 28, 56):
            demand_feats.append(_row_lag(y_hist, lag))
        for window in (7, 14, 28, 56):
            m, sd = _row_rolling_mean_std(y_hist, window)
            demand_feats.extend([m, sd])
        recent_28 = y_hist[:, -28:] if y_hist.shape[1] >= 28 else y_hist
        recent_56 = y_hist[:, -56:] if y_hist.shape[1] >= 56 else y_hist
        demand_feats.extend([
            np.count_nonzero(recent_28, axis=1, keepdims=True).astype(np.float32),
            np.count_nonzero(recent_56, axis=1, keepdims=True).astype(np.float32),
            y_hist.sum(axis=1, keepdims=True).astype(np.float32),
        ])

        feature_blocks = [np.concatenate(demand_feats, axis=1)]

        if x_hist.size and x_hist.shape[-1] > 0:
            latest_hist = x_hist[:, -1, :]
            hist_mean_7 = x_hist[:, -7:, :].mean(axis=1) if x_hist.shape[1] >= 7 else x_hist.mean(axis=1)
            hist_mean_28 = x_hist[:, -28:, :].mean(axis=1) if x_hist.shape[1] >= 28 else x_hist.mean(axis=1)
            feature_blocks.extend([latest_hist, hist_mean_7, hist_mean_28])

        if x_future.size and x_future.shape[-1] > 0:
            feature_blocks.append(x_future.reshape(N, -1))

        if "static_feat" in s:
            static_feat = np.asarray(s["static_feat"], dtype=np.float32)
            if static_feat.ndim == 2 and static_feat.shape[0] == N and static_feat.shape[1] > 0:
                feature_blocks.append(static_feat)

        origin_time = float(s.get("meta", {}).get("origin_time", 0.0))
        feature_blocks.append(np.full((N, 1), origin_time, dtype=np.float32))

        X_i = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
        X_parts.append(np.nan_to_num(X_i, nan=0.0, posinf=0.0, neginf=0.0))

    if not X_parts:
        raise ValueError("No samples were provided for bottom-level feature construction.")
    return np.vstack(X_parts).astype(np.float32), child_counts


def build_bottom_level_global_matrix(samples: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build child-level features and child-level multi-horizon targets.
    Requires samples generated with prepare_m5_experiments.CACHE_VERSION >= v5.
    """
    if not samples:
        raise ValueError("No training samples were provided to the global bottom-up benchmark.")
    missing = [i for i, s in enumerate(samples[:5]) if "y_child_target" not in s]
    if missing:
        raise KeyError(
            "Training samples do not contain 'y_child_target'. Rebuild the M5 cache with the updated "
            "prepare_m5_experiments.py, or run with --force-rebuild-cache."
        )
    X, child_counts = build_bottom_level_global_features(samples)
    Y = np.vstack([np.asarray(s["y_child_target"], dtype=np.float32) for s in samples]).astype(np.float32)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Bottom-level X/Y row mismatch: X={X.shape}, Y={Y.shape}")
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, Y, child_counts


# ============================================================
# 3) Residual quantile calibration
# ============================================================

class ResidualQuantileCalibrator:
    """
    Distribution-free residual quantile calibrator by horizon.
    For each horizon h and quantile q, estimate quantile of residuals:
        r_h = y_h - yhat_h
    and return:
        qhat_h(q) = yhat_h + quantile_q(r_h)
    """

    def __init__(self, quantiles: Sequence[float]):
        self.quantiles = list(quantiles)
        self.resid_quantiles_: Optional[np.ndarray] = None  # [H, Q]

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ResidualQuantileCalibrator":
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        residuals = y_true - y_pred  # [N, H]
        H = residuals.shape[1]
        rq = np.zeros((H, len(self.quantiles)), dtype=np.float32)
        for h in range(H):
            rq[h, :] = np.quantile(residuals[:, h], self.quantiles)
        self.resid_quantiles_ = rq
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if self.resid_quantiles_ is None:
            raise RuntimeError("Calibrator is not fitted.")
        # y_pred: [N, H]
        return y_pred[:, :, None] + self.resid_quantiles_[None, :, :]


# ============================================================
# 4) Benchmark base classes
# ============================================================

class BenchmarkBase:
    def fit(self, train_samples: Sequence[Dict[str, Any]]) -> "BenchmarkBase":
        raise NotImplementedError

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError

    def fit_quantile(self, train_samples: Sequence[Dict[str, Any]], quantiles: Sequence[float]) -> "BenchmarkBase":
        # default: residual calibration on top of point model
        self.fit(train_samples)
        y_train = self._extract_targets(train_samples)
        yhat_train = self.predict_point(train_samples)
        self.calibrator_ = ResidualQuantileCalibrator(quantiles).fit(y_train, yhat_train)
        self.quantiles_ = list(quantiles)
        return self

    def predict_quantiles(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("Call fit_quantile() before predict_quantiles().")
        yhat = self.predict_point(samples)
        return self.calibrator_.predict(yhat)

    @staticmethod
    def _extract_targets(samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        return np.vstack([s["y_target"].astype(np.float32) for s in samples])


class SklearnMultiOutputBenchmark(BenchmarkBase):
    def __init__(self, feature_builder, base_regressor: BaseEstimator):
        self.feature_builder = feature_builder
        self.base_regressor = base_regressor
        self.model_: Optional[RegressorMixin] = None

    def fit(self, train_samples: Sequence[Dict[str, Any]]) -> "SklearnMultiOutputBenchmark":
        X, Y = self.feature_builder(train_samples)
        self.model_ = clone(self.base_regressor)
        # HistGradientBoostingRegressor can stall in some environments when OpenMP
        # over-subscribes threads. Limiting sklearn/OpenMP thread pools here keeps
        # the benchmark stage reliable without changing benchmark definitions.
        with threadpool_limits(limits=1):
            self.model_.fit(X, Y)
        return self

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        X, _ = self.feature_builder(samples)
        with threadpool_limits(limits=1):
            return self.model_.predict(X).astype(np.float32)


class SeasonalNaiveAggregateBenchmark(BenchmarkBase):
    def __init__(self, seasonality: int = 7):
        self.seasonality = seasonality

    def fit(self, train_samples: Sequence[Dict[str, Any]]) -> "SeasonalNaiveAggregateBenchmark":
        return self

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        preds = []
        for s in samples:
            agg_y = s["y_hist"].sum(axis=0).reshape(-1)
            horizon = s["y_target"].shape[0]
            preds.append(_seasonal_repeat(agg_y, horizon, self.seasonality))
        return np.vstack(preds).astype(np.float32)


class BottomUpSeasonalNaiveBenchmark(BenchmarkBase):
    def __init__(self, seasonality: int = 7):
        self.seasonality = seasonality

    def fit(self, train_samples: Sequence[Dict[str, Any]]) -> "BottomUpSeasonalNaiveBenchmark":
        return self

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        preds = []
        for s in samples:
            y_child = s["y_hist"][:, :, 0]  # [N, T]
            H = s["y_target"].shape[0]
            child_preds = np.stack([
                _seasonal_repeat(y_child[i], H, self.seasonality)
                for i in range(y_child.shape[0])
            ], axis=0)
            preds.append(child_preds.sum(axis=0))
        return np.vstack(preds).astype(np.float32)


class BottomUpGlobalBoostingBenchmark(BenchmarkBase):
    """
    Global bottom-up benchmark trained at the child store-SKU level.

    A single multi-output gradient-boosting model is fitted on child/store-SKU rows
    pooled across all aggregate units and origins. At evaluation time, child-level
    multi-horizon predictions are summed to obtain aggregate forecasts.
    """

    def __init__(
        self,
        base_regressor: Optional[BaseEstimator] = None,
        random_state: int = 42,
        max_child_train_rows: Optional[int] = None,
        clip_predictions: bool = True,
    ):
        self.base_regressor = base_regressor or make_histgb_multioutput(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state,
        )
        self.random_state = int(random_state)
        self.max_child_train_rows = max_child_train_rows
        self.clip_predictions = bool(clip_predictions)
        self.model_: Optional[RegressorMixin] = None
        self.n_child_train_rows_: Optional[int] = None

    def fit(self, train_samples: Sequence[Dict[str, Any]]) -> "BottomUpGlobalBoostingBenchmark":
        X, Y, _ = build_bottom_level_global_matrix(train_samples)
        self.n_child_train_rows_ = int(X.shape[0])

        if self.max_child_train_rows is not None and X.shape[0] > int(self.max_child_train_rows):
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(X.shape[0], size=int(self.max_child_train_rows), replace=False))
            X = X[idx]
            Y = Y[idx]

        self.model_ = clone(self.base_regressor)
        with threadpool_limits(limits=1):
            self.model_.fit(X, Y)
        return self

    def predict_point(self, samples: Sequence[Dict[str, Any]]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        X, child_counts = build_bottom_level_global_features(samples)
        with threadpool_limits(limits=1):
            child_pred = np.asarray(self.model_.predict(X), dtype=np.float32)
        if child_pred.ndim == 1:
            child_pred = child_pred.reshape(-1, 1)
        if self.clip_predictions:
            child_pred = np.maximum(child_pred, 0.0)

        preds = []
        offset = 0
        for n_child in child_counts:
            block = child_pred[offset: offset + n_child]
            preds.append(block.sum(axis=0))
            offset += n_child
        return np.vstack(preds).astype(np.float32)


# ============================================================
# 5) Factory functions for benchmarks
# ============================================================

def make_elastic_net_multioutput(alpha: float = 0.001, l1_ratio: float = 0.5, random_state: int = 42):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MultiOutputRegressor(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=random_state)
        )),
    ])
    return base


def make_histgb_multioutput(max_depth: int = 6, learning_rate: float = 0.05, max_iter: int = 300, random_state: int = 42):
    base = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
        )
    )
    return base


def build_benchmark_suite(random_state: int = 42) -> Dict[str, BenchmarkBase]:
    return {
        # direct aggregate-level baselines
        "SeasonalNaive": SeasonalNaiveAggregateBenchmark(seasonality=7),
        "AggregateElasticNet": SklearnMultiOutputBenchmark(
            feature_builder=build_direct_aggregate_matrix,
            base_regressor=make_elastic_net_multioutput(alpha=0.001, l1_ratio=0.5, random_state=random_state),
        ),
        "AggregateHistGB": SklearnMultiOutputBenchmark(
            feature_builder=build_direct_aggregate_matrix,
            base_regressor=make_histgb_multioutput(max_depth=6, learning_rate=0.05, max_iter=300, random_state=random_state),
        ),
        # handcrafted cross-level baselines
        "ChildSummaryElasticNet": SklearnMultiOutputBenchmark(
            feature_builder=build_child_summary_matrix,
            base_regressor=make_elastic_net_multioutput(alpha=0.001, l1_ratio=0.5, random_state=random_state),
        ),
        "ChildSummaryHistGB": SklearnMultiOutputBenchmark(
            feature_builder=build_child_summary_matrix,
            base_regressor=make_histgb_multioutput(max_depth=6, learning_rate=0.05, max_iter=300, random_state=random_state),
        ),
        # bottom-up baseline: train globally at the store-SKU level, then aggregate upward
        "BottomUpGlobalHistGB": BottomUpGlobalBoostingBenchmark(random_state=random_state),
    }


# ============================================================
# 6) Evaluation runners
# ============================================================


def annotate_series_rows(
    series_records: Sequence[Dict[str, Any]],
    task: str,
    model: str,
    fold: int,
    metric_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    series_rows: List[Dict[str, Any]] = []
    for rec in series_records:
        row = {
            "task": task,
            "fold": int(fold),
            "model": model,
            "base_model": model,
            "agg_id": str(rec.get("agg_id", "")),
            "origin_time": int(rec.get("origin_time", -1)),
            "row_id": int(rec.get("row_id", -1)),
            "revenue_28": float(rec.get("revenue_28", np.nan)),
            "weight": float(rec.get("weight", np.nan)),
            "metric": metric_name,
        }
        if metric_name == "rmsse":
            row["metric_value"] = float(rec.get("rmsse", np.nan))
        elif metric_name == "mean_spl":
            spl_losses = rec.get("spl_losses", [])
            row["metric_value"] = float(np.mean(spl_losses)) if spl_losses else float(rec.get("mean_spl", np.nan))
        else:
            row["metric_value"] = float(rec.get(metric_name, np.nan))
        rows.append(row)
    return rows


def run_point_benchmark(
    benchmark: BenchmarkBase,
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    agg_history_store: Dict[str, pd.DataFrame],
    task: str,
    horizon: int = 28,
) -> Dict[str, Any]:
    if compute_wrmsse_from_samples is None:
        raise ImportError("compute_wrmsse_from_samples could not be imported from compute_m5_metrics.py")

    benchmark.fit(train_samples)
    preds = benchmark.predict_point(eval_samples)

    wrmsse_out = compute_wrmsse_from_samples(
        samples_raw=eval_samples,
        preds=preds,
        agg_history_store=agg_history_store,
        task=task,
        horizon=horizon,
    )

    out = {
        "preds": preds,
        "wrmsse": wrmsse_out["wrmsse"],
        "origin_wrmsse": wrmsse_out["origin_wrmsse"],
        "metric_detail": wrmsse_out,
        "series_rows": annotate_series_rows(wrmsse_out.get("series_records", []), task=task, model="", fold=-1, metric_name="rmsse"),
    }

    # Optional simple WAPE from sample targets
    y_true = np.vstack([s["y_target"] for s in eval_samples]).astype(np.float32)
    out["wape"] = float(np.sum(np.abs(preds - y_true)) / (np.sum(np.abs(y_true)) + 1e-8))
    return out


def run_quantile_benchmark(
    benchmark: BenchmarkBase,
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    agg_history_store: Dict[str, pd.DataFrame],
    task: str,
    quantiles: Sequence[float],
    horizon: int = 28,
) -> Dict[str, Any]:
    if compute_wspl_from_samples is None:
        raise ImportError("compute_wspl_from_samples could not be imported from compute_m5_metrics.py")

    benchmark.fit_quantile(train_samples, quantiles)
    pred_q = benchmark.predict_quantiles(eval_samples)

    wspl_out = compute_wspl_from_samples(
        samples_raw=eval_samples,
        pred_quantiles=pred_q,
        quantiles=quantiles,
        agg_history_store=agg_history_store,
        task=task,
        horizon=horizon,
    )

    return {
        "pred_quantiles": pred_q,
        "wspl": wspl_out["wspl"],
        "origin_wspl": wspl_out["origin_wspl"],
        "quantile_pinball": wspl_out["quantile_pinball"],
        "quantile_spl": wspl_out.get("quantile_spl", {}),
        "metric_detail": wspl_out,
        "series_rows": annotate_series_rows(wspl_out.get("series_records", []), task=task, model="", fold=-1, metric_name="mean_spl"),
    }


# ============================================================
# 7) Rolling-fold and holdout experiment runners
# ============================================================

def run_benchmark_suite_on_task(
    benchmark_suite: Dict[str, BenchmarkBase],
    task_name: str,
    task_obj: Dict[str, Any],
    agg_history_store: Dict[str, pd.DataFrame],
    quantiles: Optional[Sequence[float]] = None,
    mode: str = "rolling",
    return_series_rows: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    mode:
      - 'rolling' : evaluate across task_obj['rolling_fold_samples']
      - 'holdout' : fit on final_train_samples, evaluate on test_samples
    """
    rows: List[Dict[str, Any]] = []
    series_rows: List[Dict[str, Any]] = []

    if mode == "rolling":
        fold_samples = task_obj["rolling_fold_samples"]
        for fold in fold_samples:
            fold_id = fold["fold_id"]
            train_samples = fold["train_samples"]
            valid_samples = fold["valid_samples"]

            for model_name, bench in benchmark_suite.items():
                result = run_point_benchmark(
                    benchmark=clone_benchmark(bench),
                    train_samples=train_samples,
                    eval_samples=valid_samples,
                    agg_history_store=agg_history_store,
                    task=task_name,
                    horizon=valid_samples[0]["y_target"].shape[0],
                )
                row = {
                    "task": task_name,
                    "model": model_name,
                    "fold": fold_id,
                    "wrmsse": result["wrmsse"],
                    "wape": result["wape"],
                }
                point_series_rows = result.get("series_rows", [])
                for rec in point_series_rows:
                    rec["task"] = task_name
                    rec["model"] = model_name
                    rec["base_model"] = model_name
                    rec["fold"] = fold_id
                series_rows.extend(point_series_rows)
                rows.append(row)

                if quantiles is not None:
                    q_result = run_quantile_benchmark(
                        benchmark=clone_benchmark(bench),
                        train_samples=train_samples,
                        eval_samples=valid_samples,
                        agg_history_store=agg_history_store,
                        task=task_name,
                        quantiles=quantiles,
                        horizon=valid_samples[0]["y_target"].shape[0],
                    )
                    row_q = {
                        "task": task_name,
                        "model": model_name,
                        "fold": fold_id,
                        "wspl": q_result["wspl"],
                    }
                    q_series_rows = q_result.get("series_rows", [])
                    for rec in q_series_rows:
                        rec["task"] = task_name
                        rec["model"] = model_name
                        rec["base_model"] = model_name
                        rec["fold"] = fold_id
                    series_rows.extend(q_series_rows)
                    rows.append(row_q)

    elif mode == "holdout":
        train_samples = task_obj["final_train_samples"]
        test_samples = task_obj["test_samples"]

        for model_name, bench in benchmark_suite.items():
            result = run_point_benchmark(
                benchmark=clone_benchmark(bench),
                train_samples=train_samples,
                eval_samples=test_samples,
                agg_history_store=agg_history_store,
                task=task_name,
                horizon=test_samples[0]["y_target"].shape[0],
            )
            row = {
                "task": task_name,
                "model": model_name,
                "fold": 0,
                "wrmsse": result["wrmsse"],
                "wape": result["wape"],
            }
            point_series_rows = result.get("series_rows", [])
            for rec in point_series_rows:
                rec["task"] = task_name
                rec["model"] = model_name
                rec["base_model"] = model_name
                rec["fold"] = 0
            series_rows.extend(point_series_rows)
            rows.append(row)

            if quantiles is not None:
                q_result = run_quantile_benchmark(
                    benchmark=clone_benchmark(bench),
                    train_samples=train_samples,
                    eval_samples=test_samples,
                    agg_history_store=agg_history_store,
                    task=task_name,
                    quantiles=quantiles,
                    horizon=test_samples[0]["y_target"].shape[0],
                )
                row_q = {
                    "task": task_name,
                    "model": model_name,
                    "fold": 0,
                    "wspl": q_result["wspl"],
                }
                q_series_rows = q_result.get("series_rows", [])
                for rec in q_series_rows:
                    rec["task"] = task_name
                    rec["model"] = model_name
                    rec["base_model"] = model_name
                    rec["fold"] = 0
                series_rows.extend(q_series_rows)
                rows.append(row_q)
    else:
        raise ValueError("mode must be 'rolling' or 'holdout'.")

    rows_df = pd.DataFrame(rows)
    series_df = pd.DataFrame(series_rows)
    if return_series_rows:
        return rows_df, series_df
    return rows_df


def clone_benchmark(benchmark: BenchmarkBase) -> BenchmarkBase:
    """
    Lightweight cloning for benchmarks.
    """
    if isinstance(benchmark, SeasonalNaiveAggregateBenchmark):
        return SeasonalNaiveAggregateBenchmark(seasonality=benchmark.seasonality)
    if isinstance(benchmark, BottomUpSeasonalNaiveBenchmark):
        return BottomUpSeasonalNaiveBenchmark(seasonality=benchmark.seasonality)
    if isinstance(benchmark, BottomUpGlobalBoostingBenchmark):
        return BottomUpGlobalBoostingBenchmark(
            base_regressor=clone(benchmark.base_regressor),
            random_state=benchmark.random_state,
            max_child_train_rows=benchmark.max_child_train_rows,
            clip_predictions=benchmark.clip_predictions,
        )
    if isinstance(benchmark, SklearnMultiOutputBenchmark):
        return SklearnMultiOutputBenchmark(
            feature_builder=benchmark.feature_builder,
            base_regressor=clone(benchmark.base_regressor),
        )
    raise TypeError(f"Unsupported benchmark type: {type(benchmark)}")


# ============================================================
# 8) Result summaries for paper tables
# ============================================================

def summarize_point_results(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    point_df = df.loc[df["wrmsse"].notna()].copy()
    summary = (
        point_df.groupby(["task", "model"], as_index=False)
        .agg(
            wrmsse_mean=("wrmsse", "mean"),
            wrmsse_std=("wrmsse", "std"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
            n_folds=("fold", "count"),
        )
    )
    return summary


def summarize_quantile_results(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    q_df = df.loc[df["wspl"].notna()].copy()
    summary = (
        q_df.groupby(["task", "model"], as_index=False)
        .agg(
            wspl_mean=("wspl", "mean"),
            wspl_std=("wspl", "std"),
            n_folds=("fold", "count"),
        )
    )
    return summary


def build_main_point_table(summary_df: pd.DataFrame, metric_col: str = "wrmsse_mean") -> pd.DataFrame:
    table = summary_df.pivot(index="model", columns="task", values=metric_col)
    table = table.copy()
    table["Mean"] = table.mean(axis=1)
    table = table.sort_values("Mean")
    return table


def build_main_quantile_table(summary_df: pd.DataFrame, metric_col: str = "wspl_mean") -> pd.DataFrame:
    table = summary_df.pivot(index="model", columns="task", values=metric_col)
    table = table.copy()
    table["Mean"] = table.mean(axis=1)
    table = table.sort_values("Mean")
    return table


# ============================================================
# 9) Example usage (pseudo-main)
# ============================================================

def example_usage():
    """
    Example sketch (not executed here):

    from prepare_m5_experiments import (
        M5ExperimentConfig,
        read_m5_raw,
        build_m5_experiment_samples,
        build_m5_item_store_panel,
    )
    from compute_m5_metrics import build_task_aggregate_daily, prepare_aggregate_history_store

    cfg = M5ExperimentConfig(data_dir='./m5_data')
    raw = read_m5_raw(cfg.data_dir)
    experiment = build_m5_experiment_samples(raw, cfg)

    panel = build_m5_item_store_panel(raw['sales'], raw['calendar'], raw['prices'], cfg)

    all_rows = []
    suite = build_benchmark_suite()

    for task in cfg.tasks:
        from compute_m5_metrics import build_task_aggregate_daily, prepare_aggregate_history_store
        agg_daily = build_task_aggregate_daily(panel, task)
        hist_store = prepare_aggregate_history_store(agg_daily)

        task_rows = run_benchmark_suite_on_task(
            benchmark_suite=suite,
            task_name=task,
            task_obj=experiment[task],
            agg_history_store=hist_store,
            quantiles=[0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995],
            mode='rolling',
        )
        all_rows.append(task_rows)

    all_rows = pd.concat(all_rows, axis=0, ignore_index=True)
    point_summary = summarize_point_results(all_rows.to_dict('records'))
    point_table = build_main_point_table(point_summary)
    print(point_table)
    """
    pass


if __name__ == "__main__":
    print("m5_benchmarks.py loaded successfully.")
