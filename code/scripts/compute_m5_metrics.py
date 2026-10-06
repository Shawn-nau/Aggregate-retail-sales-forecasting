from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from prepare_m5_experiments import (
        M5ExperimentConfig,
        read_m5_raw,
        build_m5_item_store_panel,
        add_task_aggregate_id,
    )
except Exception:
    M5ExperimentConfig = None
    read_m5_raw = None
    build_m5_item_store_panel = None
    add_task_aggregate_id = None


def _pinball_loss(y: np.ndarray, qhat: np.ndarray, tau: float) -> np.ndarray:
    err = y - qhat
    return np.maximum(tau * err, (tau - 1.0) * err)


def build_task_aggregate_daily(panel_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if "agg_id" not in panel_df.columns:
        if add_task_aggregate_id is None:
            raise ImportError("add_task_aggregate_id could not be imported from prepare_m5_experiments.py")
        panel_df = add_task_aggregate_id(panel_df, task)

    out = (
        panel_df.groupby(["agg_id", "d_int"], as_index=False)
        .agg(
            y=("y", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values(["agg_id", "d_int"])
        .reset_index(drop=True)
    )
    return out


def prepare_aggregate_history_store(agg_daily_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    store: Dict[str, pd.DataFrame] = {}
    for agg_id, gdf in agg_daily_df.groupby("agg_id"):
        store[str(agg_id)] = gdf.sort_values("d_int").reset_index(drop=True).copy()
    return store


def _trim_active_history(y_hist: np.ndarray) -> np.ndarray:
    """Trim leading zeros so scaling starts at the first non-zero sale, as in M5."""
    y = np.asarray(y_hist, dtype=float)
    nz = np.nonzero(y)[0]
    if len(nz) == 0:
        return np.asarray([], dtype=float)
    return y[nz[0]:]


def _compute_rmsse_scale(y_hist: np.ndarray) -> float:
    # M5 RMSSE denominator: mean squared first difference on the active-sales portion
    y_trim = _trim_active_history(y_hist)
    if len(y_trim) <= 1:
        return 1.0
    diffs = np.diff(y_trim)
    denom = np.mean(diffs ** 2)
    return float(max(denom, 1e-8))


def _compute_spl_scale(y_hist: np.ndarray) -> float:
    # M5 SPL denominator: mean absolute first difference on the active-sales portion
    y_trim = _trim_active_history(y_hist)
    if len(y_trim) <= 1:
        return 1.0
    diffs = np.diff(y_trim)
    denom = np.mean(np.abs(diffs))
    return float(max(denom, 1e-8))


def _history_before_origin(agg_df: pd.DataFrame, origin_time: int) -> pd.DataFrame:
    return agg_df.loc[agg_df["d_int"] <= origin_time].sort_values("d_int").reset_index(drop=True)


def _last_28_before_origin(agg_df: pd.DataFrame, origin_time: int) -> pd.DataFrame:
    hist = _history_before_origin(agg_df, origin_time)
    if hist.empty:
        return hist
    return hist.tail(28).copy()


def compute_wrmsse_from_samples(
    samples_raw: Sequence[Dict[str, Any]],
    preds: np.ndarray,
    agg_history_store: Dict[str, pd.DataFrame],
    task: str,
    horizon: int = 28,
) -> Dict[str, Any]:
    if preds.ndim != 2 or preds.shape[1] != horizon:
        raise ValueError(f"preds must have shape [N, {horizon}]")

    if len(samples_raw) != preds.shape[0]:
        raise ValueError("samples_raw and preds length mismatch")

    origin_records: List[Dict[str, Any]] = []
    series_records: List[Dict[str, Any]] = []

    # static weights by origin, normalized within each origin
    for i, sample in enumerate(samples_raw):
        meta = sample["meta"]
        agg_id = str(meta["agg_id"])
        origin_time = int(meta["origin_time"])
        y_true = np.asarray(sample["y_target"], dtype=float)
        y_pred = np.asarray(preds[i], dtype=float)

        agg_df = agg_history_store[agg_id]
        hist_df = _history_before_origin(agg_df, origin_time)
        y_hist = hist_df["y"].to_numpy(dtype=float)
        scale = _compute_rmsse_scale(y_hist)

        revenue_28 = _last_28_before_origin(agg_df, origin_time)["revenue"].sum()
        rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)

        origin_records.append({
            "row_id": i,
            "origin_time": origin_time,
            "agg_id": agg_id,
            "revenue_28": float(revenue_28),
            "rmsse": float(rmsse),
        })

    origin_df = pd.DataFrame(origin_records)
    wrmsse_by_origin = {}
    for origin_time, gdf in origin_df.groupby("origin_time"):
        weights = gdf["revenue_28"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        wrmsse_origin = float(np.sum(weights * gdf["rmsse"].to_numpy(dtype=float)))
        wrmsse_by_origin[int(origin_time)] = wrmsse_origin

        tmp = gdf.copy()
        tmp["weight"] = weights
        series_records.extend(tmp.to_dict("records"))

    overall = float(np.mean(list(wrmsse_by_origin.values()))) if wrmsse_by_origin else np.nan
    return {
        "task": task,
        "wrmsse": overall,
        "origin_wrmsse": wrmsse_by_origin,
        "series_records": series_records,
    }


def compute_wspl_from_samples(
    samples_raw: Sequence[Dict[str, Any]],
    pred_quantiles: np.ndarray,
    quantiles: Sequence[float],
    agg_history_store: Dict[str, pd.DataFrame],
    task: str,
    horizon: int = 28,
) -> Dict[str, Any]:
    quantiles = list(quantiles)
    Q = len(quantiles)

    if pred_quantiles.ndim != 3 or pred_quantiles.shape[1:] != (horizon, Q):
        raise ValueError(f"pred_quantiles must have shape [N, {horizon}, {Q}]")

    if len(samples_raw) != pred_quantiles.shape[0]:
        raise ValueError("samples_raw and pred_quantiles length mismatch")

    records: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples_raw):
        meta = sample["meta"]
        agg_id = str(meta["agg_id"])
        origin_time = int(meta["origin_time"])
        y_true = np.asarray(sample["y_target"], dtype=float)  # [H]
        y_q = np.asarray(pred_quantiles[i], dtype=float)       # [H,Q]

        agg_df = agg_history_store[agg_id]
        revenue_28 = _last_28_before_origin(agg_df, origin_time)["revenue"].sum()

        hist_df = _history_before_origin(agg_df, origin_time)
        y_hist = hist_df["y"].to_numpy(dtype=float)
        spl_scale = _compute_spl_scale(y_hist)

        pinball_losses = []
        spl_losses = []
        for q_idx, tau in enumerate(quantiles):
            pinball = float(_pinball_loss(y_true, y_q[:, q_idx], tau).mean())
            spl = float(pinball / spl_scale)
            pinball_losses.append(pinball)
            spl_losses.append(spl)

        records.append({
            "row_id": i,
            "origin_time": origin_time,
            "agg_id": agg_id,
            "revenue_28": float(revenue_28),
            "spl_scale": float(spl_scale),
            "q_losses": pinball_losses,
            "spl_losses": spl_losses,
            "mean_spl": float(np.mean(spl_losses)),
        })

    rec_df = pd.DataFrame(records)

    wspl_by_origin = {}
    quantile_pinball = {float(tau): [] for tau in quantiles}
    quantile_spl = {float(tau): [] for tau in quantiles}
    series_records: List[Dict[str, Any]] = []

    for origin_time, gdf in rec_df.groupby("origin_time"):
        weights = gdf["revenue_28"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()

        pinball_mat = np.stack(gdf["q_losses"].to_list(), axis=0)  # [n_series, Q]
        spl_mat = np.stack(gdf["spl_losses"].to_list(), axis=0)    # [n_series, Q]

        weighted_pinball = (weights[:, None] * pinball_mat).sum(axis=0)
        weighted_spl = (weights[:, None] * spl_mat).sum(axis=0)
        wspl_origin = float(weighted_spl.mean())
        wspl_by_origin[int(origin_time)] = wspl_origin

        for q_idx, tau in enumerate(quantiles):
            quantile_pinball[float(tau)].append(float(weighted_pinball[q_idx]))
            quantile_spl[float(tau)].append(float(weighted_spl[q_idx]))

        tmp = gdf.copy()
        tmp["weight"] = weights
        series_records.extend(tmp.to_dict("records"))

    overall = float(np.mean(list(wspl_by_origin.values()))) if wspl_by_origin else np.nan
    quantile_pinball_mean = {tau: float(np.mean(vals)) for tau, vals in quantile_pinball.items()}
    quantile_spl_mean = {tau: float(np.mean(vals)) for tau, vals in quantile_spl.items()}

    return {
        "task": task,
        "wspl": overall,
        "origin_wspl": wspl_by_origin,
        "quantile_pinball": quantile_pinball_mean,
        "quantile_spl": quantile_spl_mean,
        "series_records": series_records,
    }

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
    table = summary_df.pivot(index="model", columns="task", values=metric_col).copy()
    table["Mean"] = table.mean(axis=1)
    return table.sort_values("Mean")


def build_main_quantile_table(summary_df: pd.DataFrame, metric_col: str = "wspl_mean") -> pd.DataFrame:
    table = summary_df.pivot(index="model", columns="task", values=metric_col).copy()
    table["Mean"] = table.mean(axis=1)
    return table.sort_values("Mean")


if __name__ == "__main__":
    print("compute_m5_metrics.py loaded successfully.")