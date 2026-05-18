from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class M5ExperimentConfig:
    data_dir: str
    t_hist: int = 364
    horizon: int = 28
    tasks: Tuple[str, ...] = ("store_dept", "store_cat", "state_dept")
    valid_size: int = 4
    test_size: int = 4
    internal_valid_size: int = 3
    gap: int = 0
    min_train_origins: int = 20
    min_final_train_origins: int = 20
    fill_value: float = 0.0
    sku_universe_mode: str = "history"
    use_sell_price: bool = True
    use_price_change: bool = True
    use_snap: bool = True
    use_events: bool = True
    use_calendar_basic: bool = True
    keep_only_active_items: bool = False
    store_sku_sample_frac: float = 1.0
    store_sku_sample_seed: int = 42


CACHE_VERSION = "v5_child_targets_bottomup_global_boosting"


class ProgressPrinter:
    def __init__(self, enabled: bool = True, min_interval_sec: float = 5.0) -> None:
        self.enabled = enabled
        self.min_interval_sec = min_interval_sec
        self.global_start = time.time()
        self.stage_start = time.time()
        self.last_print = 0.0

    def stage(self, msg: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        self.stage_start = now
        self.last_print = now
        print(f"[M5-PREP] {msg}", flush=True)

    def update(self, current: int, total: int, prefix: str) -> None:
        if not self.enabled or total <= 0:
            return
        now = time.time()
        if current < total and (now - self.last_print) < self.min_interval_sec:
            return
        elapsed = now - self.stage_start
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (total - current) / rate if rate > 0 else math.inf
        eta_text = "?" if not np.isfinite(eta) else _format_seconds(eta)
        print(
            f"[M5-PREP] {prefix}: {current}/{total} ({100*current/total:5.1f}%) | "
            f"elapsed {_format_seconds(elapsed)} | ETA {eta_text}",
            flush=True,
        )
        self.last_print = now

    def done(self, msg: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        print(
            f"[M5-PREP] {msg} | stage {_format_seconds(now-self.stage_start)} | total {_format_seconds(now-self.global_start)}",
            flush=True,
        )
        self.last_print = now


def _format_seconds(seconds: float) -> str:
    seconds = int(max(0, round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _cfg_hash(cfg: M5ExperimentConfig) -> str:
    return hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def default_cache_dir(cfg: M5ExperimentConfig) -> str:
    return os.path.join(cfg.data_dir, "_m5_cache", f"{CACHE_VERSION}_{_cfg_hash(cfg)}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_pickle(obj: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def read_m5_raw(data_dir: str) -> Dict[str, pd.DataFrame]:
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    if not os.path.exists(sales_path):
        sales_path = os.path.join(data_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")
    price_path = os.path.join(data_dir, "sell_prices.csv")

    if not os.path.exists(sales_path):
        raise FileNotFoundError("Cannot find sales_train_evaluation.csv or sales_train_validation.csv")
    if not os.path.exists(calendar_path):
        raise FileNotFoundError("Cannot find calendar.csv")
    if not os.path.exists(price_path):
        raise FileNotFoundError("Cannot find sell_prices.csv")

    header = pd.read_csv(sales_path, nrows=0)
    day_cols = [c for c in header.columns if c.startswith("d_")]
    sales_dtypes = {
        "id": "category",
        "item_id": "category",
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
    }
    sales_dtypes.update({c: np.int16 for c in day_cols})
    sales = pd.read_csv(sales_path, dtype=sales_dtypes)

    cal_dtypes = {
        "wm_yr_wk": np.int16,
        "wday": np.int8,
        "month": np.int8,
        "year": np.int16,
        "snap_CA": np.int8,
        "snap_TX": np.int8,
        "snap_WI": np.int8,
        "weekday": "category",
        "event_name_1": "category",
        "event_type_1": "category",
        "event_name_2": "category",
        "event_type_2": "category",
    }
    calendar = pd.read_csv(calendar_path, dtype=cal_dtypes, parse_dates=["date"])
    prices = pd.read_csv(
        price_path,
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": np.int16, "sell_price": np.float32},
    )
    return {"sales": sales, "calendar": calendar, "prices": prices}


def _apply_store_sku_sampling(panel: pd.DataFrame, cfg: M5ExperimentConfig, progress: bool = True) -> pd.DataFrame:
    sample_frac = float(cfg.store_sku_sample_frac)
    if sample_frac >= 1.0:
        return panel
    if sample_frac <= 0.0:
        raise ValueError(f"store_sku_sample_frac must be in (0, 1], got {sample_frac}")

    printer = ProgressPrinter(progress)
    printer.stage(
        f"Applying reproducible store-SKU sampling (frac={sample_frac:.4f}, seed={cfg.store_sku_sample_seed})..."
    )
    unique_skus = pd.Index(panel["sku_id"].drop_duplicates().astype(str))
    n_total = int(len(unique_skus))
    if n_total == 0:
        raise ValueError("No store-SKU units available for sampling.")

    n_keep = min(n_total, max(1, int(math.floor(n_total * sample_frac))))
    if n_keep >= n_total:
        printer.done(f"Sampling requested but all store-SKU units are retained (n={n_total})")
        return panel

    rng = np.random.default_rng(int(cfg.store_sku_sample_seed))
    keep_skus = set(rng.choice(unique_skus.to_numpy(), size=n_keep, replace=False).tolist())
    sampled = panel.loc[panel["sku_id"].astype(str).isin(keep_skus)].copy()
    sampled.sort_values(["store_id", "item_id", "d_int"], inplace=True, kind="mergesort")
    sampled.reset_index(drop=True, inplace=True)

    n_sampled = int(sampled["sku_id"].nunique(dropna=True))
    n_rows = int(len(sampled))
    printer.done(
        f"Store-SKU sampling kept {n_sampled}/{n_total} units ({n_sampled / n_total:.2%}); panel rows={n_rows}"
    )
    return sampled


def build_m5_item_store_panel(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: M5ExperimentConfig,
    cache_dir: Optional[str] = None,
    force_rebuild: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    if cache_dir is None:
        cache_dir = default_cache_dir(cfg)
    cache_path = os.path.join(cache_dir, "base_panel.pkl")
    printer = ProgressPrinter(progress)
    if os.path.exists(cache_path) and not force_rebuild:
        printer.stage("Loading cached item-store panel...")
        panel = _load_pickle(cache_path)
        printer.done(f"Loaded cached panel with shape={panel.shape}")
        return panel

    printer.stage("Building item-store panel from raw CSV tables...")
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    panel = sales[id_cols + day_cols].melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="y")
    panel["y"] = panel["y"].astype(np.float32, copy=False)

    cal = calendar.copy()
    if "d_int" not in cal.columns:
        cal["d_int"] = cal["d"].str.replace("d_", "", regex=False).astype(np.int16)
    use_cols = ["d", "date", "wm_yr_wk", "d_int"]
    if cfg.use_calendar_basic:
        use_cols += [c for c in ["wday", "month", "year"] if c in cal.columns]
    if cfg.use_snap:
        use_cols += [c for c in ["snap_CA", "snap_TX", "snap_WI"] if c in cal.columns]
    if cfg.use_events:
        use_cols += [c for c in ["event_name_1", "event_name_2", "event_type_1", "event_type_2"] if c in cal.columns]
    use_cols = list(dict.fromkeys(use_cols))
    panel = panel.merge(cal[use_cols], on="d", how="left", copy=False)

    if cfg.use_sell_price:
        panel = panel.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", copy=False)
        panel["sell_price"] = panel["sell_price"].fillna(cfg.fill_value).astype(np.float32, copy=False)
    else:
        panel["sell_price"] = np.float32(cfg.fill_value)

    panel.sort_values(["store_id", "item_id", "d_int"], inplace=True, kind="mergesort")

    if cfg.use_price_change:
        lag = panel.groupby(["store_id", "item_id"], observed=True)["sell_price"].shift(1).astype(np.float32, copy=False)
        current_price = panel["sell_price"].to_numpy(dtype=np.float32, copy=False)
        lag_price = lag.to_numpy(dtype=np.float32, copy=False)
        ratio = np.zeros(len(panel), dtype=np.float32)
        valid = lag_price > 0
        np.divide(current_price, lag_price, out=ratio, where=valid)
        ratio[valid] -= 1.0
        panel["price_change_ratio"] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        panel["price_change_ratio"] = np.float32(0.0)

    if cfg.use_snap:
        state = panel["state_id"].astype(str).to_numpy(copy=False)
        snap = np.zeros(len(panel), dtype=np.float32)
        if "snap_CA" in panel.columns:
            m = state == "CA"
            snap[m] = panel.loc[m, "snap_CA"].to_numpy(dtype=np.float32, copy=False)
        if "snap_TX" in panel.columns:
            m = state == "TX"
            snap[m] = panel.loc[m, "snap_TX"].to_numpy(dtype=np.float32, copy=False)
        if "snap_WI" in panel.columns:
            m = state == "WI"
            snap[m] = panel.loc[m, "snap_WI"].to_numpy(dtype=np.float32, copy=False)
        panel["snap_flag"] = snap
    else:
        panel["snap_flag"] = np.float32(0.0)

    if cfg.use_events:
        evt_cols = [c for c in ["event_name_1", "event_name_2"] if c in panel.columns]
        if evt_cols:
            evt = np.zeros(len(panel), dtype=bool)
            for c in evt_cols:
                evt |= panel[c].notna().to_numpy()
            panel["event_flag"] = evt.astype(np.float32)
        else:
            panel["event_flag"] = np.float32(0.0)
    else:
        panel["event_flag"] = np.float32(0.0)

    panel["date"] = pd.to_datetime(panel["date"])
    if "wday" not in panel.columns:
        panel["wday"] = (panel["date"].dt.weekday + 1).astype(np.int8)
    if "month" not in panel.columns:
        panel["month"] = panel["date"].dt.month.astype(np.int8)
    if "year" not in panel.columns:
        panel["year"] = panel["date"].dt.year.astype(np.int16)

    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        panel[c] = panel[c].astype(str)
    panel["sku_id"] = panel["item_id"] + "__" + panel["store_id"]
    panel["revenue"] = (panel["y"].to_numpy(dtype=np.float32, copy=False) * panel["sell_price"].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)

    if cfg.keep_only_active_items:
        keep = panel.groupby(["item_id", "store_id"], observed=True)["y"].sum().reset_index()
        keep = keep.loc[keep["y"] > 0, ["item_id", "store_id"]]
        panel = panel.merge(keep.assign(_keep=1), on=["item_id", "store_id"], how="inner").drop(columns=["_keep"])

    panel = _apply_store_sku_sampling(panel, cfg, progress=progress)

    for col, dt in {
        "d_int": np.int16,
        "wday": np.int8,
        "month": np.int8,
        "year": np.int16,
        "y": np.float32,
        "sell_price": np.float32,
        "price_change_ratio": np.float32,
        "snap_flag": np.float32,
        "event_flag": np.float32,
        "revenue": np.float32,
    }.items():
        if col in panel.columns:
            panel[col] = panel[col].astype(dt, copy=False)

    panel.reset_index(drop=True, inplace=True)
    _save_pickle(panel, cache_path)
    printer.done(f"Built and cached panel with shape={panel.shape}")
    return panel


def add_task_aggregate_id(panel_df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = panel_df.copy()
    if task == "store_dept":
        df["agg_id"] = df["store_id"] + "__" + df["dept_id"]
    elif task == "store_cat":
        df["agg_id"] = df["store_id"] + "__" + df["cat_id"]
    elif task == "state_dept":
        df["agg_id"] = df["state_id"] + "__" + df["dept_id"]
    elif task == "state_cat":
        df["agg_id"] = df["state_id"] + "__" + df["cat_id"]
    elif task == "store":
        df["agg_id"] = df["store_id"]
    else:
        raise ValueError(f"Unknown task: {task}")
    return df


def get_feature_columns(panel_df: pd.DataFrame) -> Dict[str, List[str]]:
    hist_feature_cols = [c for c in ["sell_price", "price_change_ratio", "snap_flag", "event_flag", "wday", "month"] if c in panel_df.columns]
    return {
        "hist_feature_cols": hist_feature_cols,
        "future_feature_cols": hist_feature_cols.copy(),
        "static_feature_cols": [],
    }


def sorted_unique_times(df: pd.DataFrame, time_col: str = "d_int") -> List[Any]:
    return list(pd.Series(df[time_col].drop_duplicates()).sort_values())


def feasible_origin_indices(num_times: int, t_hist: int, horizon: int, step: int = 1) -> List[int]:
    start = t_hist - 1
    end = num_times - horizon - 1
    if end < start:
        return []
    # Use the step parameter to jump forward by 'horizon' days
    return list(range(start, end + 1, step))


def make_holdout_origin_split(all_times: Sequence[Any], t_hist: int, horizon: int, valid_size: int, test_size: int, gap: int = 0, min_train_origins: int = 12) -> Dict[str, List[int]]:
    origins = feasible_origin_indices(len(all_times), t_hist, horizon,step=horizon)
    if len(origins) < valid_size + test_size + min_train_origins:
        raise ValueError("Not enough origins for holdout split.")
    test_origins = origins[-test_size:]
    test_start = test_origins[0]
    max_trainval_origin = test_start - gap - horizon - 1
    trainval_origins_for_test = [o for o in origins if o <= max_trainval_origin]
    if len(trainval_origins_for_test) < valid_size + min_train_origins:
        raise ValueError("Not enough trainval origins for test.")
    valid_origins = trainval_origins_for_test[-valid_size:]
    valid_start = valid_origins[0]
    max_train_origin = valid_start - gap - horizon - 1
    train_origins_for_valid = [o for o in origins if o <= max_train_origin]
    if len(train_origins_for_valid) < min_train_origins:
        raise ValueError("Not enough train origins for valid.")
    return {
        "train_origins_for_valid": train_origins_for_valid,
        "valid_origins": valid_origins,
        "trainval_origins_for_test": trainval_origins_for_test,
        "test_origins": test_origins,
    }


def make_strict_internal_valid_split(trainval_origins: Sequence[int], horizon: int, internal_valid_size: int, gap: int = 0, min_final_train_origins: int = 12) -> Dict[str, List[int]]:
    trainval_origins = list(trainval_origins)
    if len(trainval_origins) < internal_valid_size + min_final_train_origins:
        raise ValueError("Not enough trainval origins.")
    internal_valid_origins = trainval_origins[-internal_valid_size:]
    internal_valid_start = internal_valid_origins[0]
    max_final_train_origin = internal_valid_start - gap - horizon - 1
    final_train_origins = [o for o in trainval_origins if o <= max_final_train_origin]
    if len(final_train_origins) < min_final_train_origins:
        raise ValueError("Not enough final train origins.")
    return {"final_train_origins": final_train_origins, "internal_valid_origins": internal_valid_origins}


def make_expanding_window_folds(all_times: Sequence[Any], t_hist: int, horizon: int, valid_size: int, min_train_origins: int = 12, gap: int = 0, step_size: int = 1, max_folds: Optional[int] = None) -> List[Dict[str, List[int]]]:
    origins = feasible_origin_indices(len(all_times), t_hist, horizon,step=horizon)
    folds: List[Dict[str, List[int]]] = []
    start_idx = min_train_origins + gap + horizon
    for valid_start_idx in range(start_idx, len(origins) - valid_size + 1, step_size):
        valid_origins = origins[valid_start_idx: valid_start_idx + valid_size]
        valid_start = valid_origins[0]
        max_train_origin = valid_start - gap - horizon - 1
        train_origins = [o for o in origins if o <= max_train_origin]
        if len(train_origins) < min_train_origins:
            continue
        folds.append({"train_origins": train_origins, "valid_origins": valid_origins})
        if max_folds is not None and len(folds) >= max_folds:
            break
    if not folds:
        raise ValueError("No valid expanding-window folds generated.")
    return folds


def _build_task_array_store(panel_df: pd.DataFrame, task: str, cfg: M5ExperimentConfig, progress: bool = True) -> Dict[str, Any]:
    printer = ProgressPrinter(progress)
    printer.stage(f"Building fast task arrays for task={task}...")
    panel_task = add_task_aggregate_id(panel_df, task)
    feat = get_feature_columns(panel_task)
    hist_feature_cols = feat["hist_feature_cols"]
    future_feature_cols = feat["future_feature_cols"]
    static_feature_cols = feat["static_feature_cols"]
    all_times = np.asarray(sorted_unique_times(panel_task, "d_int"), dtype=np.int32)
    first_time = int(all_times[0])
    num_times = len(all_times)

    holdout_split = make_holdout_origin_split(all_times.tolist(), cfg.t_hist, cfg.horizon, cfg.valid_size, cfg.test_size, cfg.gap, cfg.min_train_origins)
    strict_inner = make_strict_internal_valid_split(holdout_split["trainval_origins_for_test"], cfg.horizon, cfg.internal_valid_size, cfg.gap, cfg.min_final_train_origins)
    rolling_folds = make_expanding_window_folds(all_times.tolist(), cfg.t_hist, cfg.horizon, cfg.valid_size, cfg.min_train_origins, cfg.gap, step_size=2, max_folds=5)

    agg_daily_df = (
        panel_task.groupby(["agg_id", "d_int"], as_index=False, observed=True)
        .agg(y=("y", "sum"), revenue=("revenue", "sum"))
        .sort_values(["agg_id", "d_int"], kind="mergesort")
        .reset_index(drop=True)
    )
    agg_daily_df["y"] = agg_daily_df["y"].astype(np.float32, copy=False)
    agg_daily_df["revenue"] = agg_daily_df["revenue"].astype(np.float32, copy=False)

    base_cols = ["agg_id", "sku_id", "d_int", "y"]
    extra_cols = [c for c in set(hist_feature_cols + future_feature_cols + static_feature_cols + ["revenue"]) if c in panel_task.columns]
    compact_df = panel_task[list(dict.fromkeys(base_cols + extra_cols))].copy()
    compact_df.sort_values(["agg_id", "sku_id", "d_int"], inplace=True, kind="mergesort")

    agg_ids = compact_df["agg_id"].drop_duplicates().astype(str).tolist()
    aggregate_arrays: Dict[str, Dict[str, Any]] = {}
    total = len(agg_ids)
    for idx, (agg_id, gdf) in enumerate(compact_df.groupby("agg_id", observed=True, sort=False), start=1):
        agg_id = str(agg_id)
        sku_ids = gdf["sku_id"].drop_duplicates().astype(str).tolist()
        sku_codes = pd.Categorical(gdf["sku_id"].astype(str), categories=sku_ids, ordered=True).codes.astype(np.int32)
        time_pos = gdf["d_int"].to_numpy(dtype=np.int32, copy=False) - first_time
        n_skus = len(sku_ids)

        y_full = np.full((n_skus, num_times), cfg.fill_value, dtype=np.float32)
        y_full[sku_codes, time_pos] = gdf["y"].to_numpy(dtype=np.float32, copy=False)

        shared_feature_matrix = None
        x_hist_full = None
        x_future_full = None
        same_features = tuple(hist_feature_cols) == tuple(future_feature_cols)
        if hist_feature_cols and same_features:
            shared_feature_matrix = np.full((n_skus, num_times, len(hist_feature_cols)), cfg.fill_value, dtype=np.float32)
            shared_feature_matrix[sku_codes, time_pos, :] = gdf[hist_feature_cols].to_numpy(dtype=np.float32, copy=False)
        else:
            if hist_feature_cols:
                x_hist_full = np.full((n_skus, num_times, len(hist_feature_cols)), cfg.fill_value, dtype=np.float32)
                x_hist_full[sku_codes, time_pos, :] = gdf[hist_feature_cols].to_numpy(dtype=np.float32, copy=False)
            if future_feature_cols:
                x_future_full = np.full((n_skus, num_times, len(future_feature_cols)), cfg.fill_value, dtype=np.float32)
                x_future_full[sku_codes, time_pos, :] = gdf[future_feature_cols].to_numpy(dtype=np.float32, copy=False)

        static_arr = None
        if static_feature_cols:
            static_arr = (
                gdf[["sku_id"] + list(static_feature_cols)]
                .drop_duplicates(subset=["sku_id"], keep="last")
                .set_index("sku_id")
                .reindex(sku_ids)
                .fillna(cfg.fill_value)
                .to_numpy(dtype=np.float32, copy=False)
            )

        agg_y_full = np.full((num_times,), cfg.fill_value, dtype=np.float32)
        gagg = agg_daily_df.loc[agg_daily_df["agg_id"].astype(str) == agg_id, ["d_int", "y"]]
        agg_y_full[gagg["d_int"].to_numpy(dtype=np.int32, copy=False) - first_time] = gagg["y"].to_numpy(dtype=np.float32, copy=False)

        aggregate_arrays[agg_id] = {
            "sku_ids": sku_ids,
            "y_full": y_full,
            "shared_feature_matrix": shared_feature_matrix,
            "x_hist_full": x_hist_full,
            "x_future_full": x_future_full,
            "static_arr": static_arr,
            "agg_y_full": agg_y_full,
        }
        printer.update(idx, total, prefix=f"task={task} aggregate tensor build")
    printer.done(f"Fast task arrays ready for task={task} with {total} aggregates")
    return {
        "task": task,
        "all_times": all_times,
        "hist_feature_cols": hist_feature_cols,
        "future_feature_cols": future_feature_cols,
        "static_feature_cols": static_feature_cols,
        "holdout_split": holdout_split,
        "strict_inner_split": strict_inner,
        "rolling_folds": rolling_folds,
        "agg_ids": agg_ids,
        "agg_daily_df": agg_daily_df,
        "aggregate_arrays": aggregate_arrays,
        "store_sku_sample_frac": float(cfg.store_sku_sample_frac),
        "store_sku_sample_seed": int(cfg.store_sku_sample_seed),
    }


def build_one_sample_from_task_store(task_store: Dict[str, Any], agg_id_value: Any, origin_idx: int, t_hist: int, horizon: int, fill_value: float = 0.0, sku_universe_mode: str = "history") -> Optional[Dict[str, Any]]:
    all_times = task_store["all_times"]
    if origin_idx < t_hist - 1 or origin_idx + horizon >= len(all_times):
        return None
    agg_id_value = str(agg_id_value)
    agg_store = task_store["aggregate_arrays"].get(agg_id_value)
    if agg_store is None:
        return None
    hist_slice = slice(origin_idx - t_hist + 1, origin_idx + 1)
    future_slice = slice(origin_idx + 1, origin_idx + 1 + horizon)
    y_hist = np.ascontiguousarray(agg_store["y_full"][:, hist_slice][:, :, None], dtype=np.float32)
    if agg_store["shared_feature_matrix"] is not None:
        x_hist = np.ascontiguousarray(agg_store["shared_feature_matrix"][:, hist_slice, :], dtype=np.float32)
        x_future = np.ascontiguousarray(agg_store["shared_feature_matrix"][:, future_slice, :], dtype=np.float32)
    else:
        x_hist = np.ascontiguousarray(agg_store["x_hist_full"][:, hist_slice, :], dtype=np.float32) if agg_store["x_hist_full"] is not None else np.zeros((y_hist.shape[0], t_hist, 0), dtype=np.float32)
        x_future = np.ascontiguousarray(agg_store["x_future_full"][:, future_slice, :], dtype=np.float32) if agg_store["x_future_full"] is not None else np.zeros((y_hist.shape[0], horizon, 0), dtype=np.float32)
    sample = {
        "y_hist": y_hist,
        "x_hist": x_hist,
        "x_future": x_future,
        "y_child_target": np.ascontiguousarray(agg_store["y_full"][:, future_slice], dtype=np.float32),
        "y_target": np.ascontiguousarray(agg_store["agg_y_full"][future_slice], dtype=np.float32),
        "meta": {
            "agg_id": agg_id_value,
            "origin_time": int(all_times[origin_idx]),
            "hist_times": all_times[hist_slice].tolist(),
            "future_times": all_times[future_slice].tolist(),
            "n_skus": int(y_hist.shape[0]),
        },
    }
    if task_store["static_feature_cols"]:
        sample["static_feat"] = np.ascontiguousarray(agg_store["static_arr"], dtype=np.float32)
    return sample


def generate_samples_from_task_store(task_store: Dict[str, Any], origin_indices: Sequence[int], t_hist: int, horizon: int, fill_value: float = 0.0, aggregate_ids: Optional[Sequence[Any]] = None, sku_universe_mode: str = "history", progress: bool = True, progress_prefix: str = "sample generation") -> List[Dict[str, Any]]:
    if aggregate_ids is None:
        aggregate_ids = task_store["agg_ids"]
    aggregate_ids = [str(x) for x in aggregate_ids]
    printer = ProgressPrinter(progress)
    printer.stage(f"{progress_prefix} started...")
    total = len(origin_indices) * len(aggregate_ids)
    done = 0
    out: List[Dict[str, Any]] = []
    for origin_idx in origin_indices:
        for agg_id in aggregate_ids:
            s = build_one_sample_from_task_store(task_store, agg_id, origin_idx, t_hist, horizon, fill_value, sku_universe_mode)
            if s is not None:
                out.append(s)
            done += 1
            printer.update(done, total, prefix=progress_prefix)
    printer.done(f"{progress_prefix} finished with {len(out)} samples")
    return out


# Backward-compatible panel helpers

def build_one_sample_from_panel(panel_df: pd.DataFrame, agg_id_value: Any, origin_idx: int, all_times: Sequence[Any], t_hist: int, horizon: int, agg_id_col: str = "agg_id", sku_id_col: str = "sku_id", time_col: str = "d_int", target_col: str = "y", hist_feature_cols: Sequence[str] = (), future_feature_cols: Sequence[str] = (), static_feature_cols: Optional[Sequence[str]] = None, fill_value: float = 0.0, sku_universe_mode: str = "history") -> Optional[Dict[str, Any]]:
    if origin_idx < t_hist - 1 or origin_idx + horizon >= len(all_times):
        return None
    hist_times = list(all_times[origin_idx - t_hist + 1: origin_idx + 1])
    future_times = list(all_times[origin_idx + 1: origin_idx + 1 + horizon])
    agg_df = panel_df.loc[panel_df[agg_id_col] == agg_id_value].copy()
    if agg_df.empty:
        return None
    hist_df = agg_df.loc[agg_df[time_col].isin(hist_times)]
    future_df = agg_df.loc[agg_df[time_col].isin(future_times)]
    if sku_universe_mode == "history":
        sku_ids = sorted(hist_df[sku_id_col].dropna().unique().tolist())
    else:
        sku_ids = sorted(pd.concat([hist_df[sku_id_col], future_df[sku_id_col]]).dropna().unique().tolist())
    if not sku_ids:
        return None
    y_hist_list, y_target_list, x_hist_list, x_future_list, static_list = [], [], [], [], []
    for sku in sku_ids:
        sku_df = agg_df.loc[agg_df[sku_id_col] == sku].sort_values(time_col).set_index(time_col)
        y_hist_arr = sku_df.reindex(hist_times)[target_col].fillna(fill_value).to_numpy(dtype=np.float32).reshape(t_hist, 1)
        y_target_arr = sku_df.reindex(future_times)[target_col].fillna(fill_value).to_numpy(dtype=np.float32)
        x_hist_arr = sku_df.reindex(hist_times)[list(hist_feature_cols)].fillna(fill_value).to_numpy(dtype=np.float32) if hist_feature_cols else np.zeros((t_hist, 0), dtype=np.float32)
        x_future_arr = sku_df.reindex(future_times)[list(future_feature_cols)].fillna(fill_value).to_numpy(dtype=np.float32) if future_feature_cols else np.zeros((horizon, 0), dtype=np.float32)
        y_hist_list.append(y_hist_arr)
        y_target_list.append(y_target_arr)
        x_hist_list.append(x_hist_arr)
        x_future_list.append(x_future_arr)
        if static_feature_cols:
            sdf = sku_df[list(static_feature_cols)].dropna(how="all")
            static_list.append(sdf.iloc[-1].fillna(fill_value).to_numpy(dtype=np.float32) if not sdf.empty else np.full((len(static_feature_cols),), fill_value, dtype=np.float32))
    future_target_series = future_df.groupby(time_col)[target_col].sum().reindex(future_times, fill_value=fill_value)
    sample = {
        "y_hist": np.stack(y_hist_list, axis=0),
        "x_hist": np.stack(x_hist_list, axis=0),
        "x_future": np.stack(x_future_list, axis=0),
        "y_child_target": np.stack(y_target_list, axis=0).astype(np.float32),
        "y_target": future_target_series.to_numpy(dtype=np.float32),
        "meta": {"agg_id": agg_id_value, "origin_time": int(all_times[origin_idx]), "hist_times": hist_times, "future_times": future_times, "n_skus": len(sku_ids)},
    }
    if static_feature_cols:
        sample["static_feat"] = np.stack(static_list, axis=0)
    return sample


def generate_samples_from_panel(panel_df: pd.DataFrame, origin_indices: Sequence[int], t_hist: int, horizon: int, agg_id_col: str = "agg_id", sku_id_col: str = "sku_id", time_col: str = "d_int", target_col: str = "y", hist_feature_cols: Sequence[str] = (), future_feature_cols: Sequence[str] = (), static_feature_cols: Optional[Sequence[str]] = None, fill_value: float = 0.0, aggregate_ids: Optional[Sequence[Any]] = None, sku_universe_mode: str = "history") -> List[Dict[str, Any]]:
    all_times = sorted_unique_times(panel_df, time_col)
    if aggregate_ids is None:
        aggregate_ids = sorted(panel_df[agg_id_col].dropna().unique().tolist())
    out: List[Dict[str, Any]] = []
    for origin_idx in origin_indices:
        for agg_id_value in aggregate_ids:
            s = build_one_sample_from_panel(panel_df, agg_id_value, origin_idx, all_times, t_hist, horizon, agg_id_col, sku_id_col, time_col, target_col, hist_feature_cols, future_feature_cols, static_feature_cols, fill_value, sku_universe_mode)
            if s is not None:
                out.append(s)
    return out


def _task_cache_dir(cache_dir: str, task: str) -> str:
    return os.path.join(cache_dir, f"task__{task}")


def _load_or_build_task_package(raw_data: Optional[Dict[str, pd.DataFrame]], task: str, cfg: M5ExperimentConfig, cache_dir: Optional[str] = None, force_rebuild: bool = False, progress: bool = True) -> Dict[str, Any]:
    if cache_dir is None:
        cache_dir = default_cache_dir(cfg)
    pkg_path = os.path.join(_task_cache_dir(cache_dir, task), "task_package.pkl")
    if os.path.exists(pkg_path) and not force_rebuild:
        printer = ProgressPrinter(progress)
        printer.stage(f"Loading cached task package for task={task}...")
        pkg = _load_pickle(pkg_path)
        printer.done(f"Loaded cached task package for task={task}")
        return pkg
    if raw_data is None:
        raw_data = read_m5_raw(cfg.data_dir)
    panel = build_m5_item_store_panel(raw_data["sales"], raw_data["calendar"], raw_data["prices"], cfg, cache_dir=cache_dir, force_rebuild=force_rebuild, progress=progress)
    pkg = _build_task_array_store(panel, task, cfg, progress=progress)
    _save_pickle(pkg, pkg_path)
    return pkg


def _load_or_build_sample_cache(task_package: Dict[str, Any], split_name: str, origin_indices: Sequence[int], cfg: M5ExperimentConfig, cache_dir: Optional[str] = None, force_rebuild: bool = False, progress: bool = True) -> List[Dict[str, Any]]:
    if cache_dir is None:
        cache_dir = default_cache_dir(cfg)
    path = os.path.join(_task_cache_dir(cache_dir, task_package["task"]), f"{split_name}.pkl")
    if os.path.exists(path) and not force_rebuild:
        printer = ProgressPrinter(progress)
        printer.stage(f"Loading cached samples: task={task_package['task']} split={split_name}...")
        obj = _load_pickle(path)
        printer.done(f"Loaded cached samples: task={task_package['task']} split={split_name} n={len(obj)}")
        return obj
    samples = generate_samples_from_task_store(task_package, origin_indices, cfg.t_hist, cfg.horizon, cfg.fill_value, task_package["agg_ids"], cfg.sku_universe_mode, progress=progress, progress_prefix=f"task={task_package['task']} split={split_name}")
    _save_pickle(samples, path)
    return samples


def build_task_package(raw_data: Optional[Dict[str, pd.DataFrame]], task: str, cfg: M5ExperimentConfig, cache_dir: Optional[str] = None, force_rebuild: bool = False, progress: bool = True) -> Dict[str, Any]:
    return _load_or_build_task_package(raw_data, task, cfg, cache_dir, force_rebuild, progress)


def build_m5_experiment_samples(raw_data: Optional[Dict[str, pd.DataFrame]], cfg: M5ExperimentConfig, need_outer: bool = True, need_holdout: bool = True, need_rolling: bool = True, cache_dir: Optional[str] = None, force_rebuild: bool = False, progress: bool = True) -> Dict[str, Any]:
    if cache_dir is None:
        cache_dir = default_cache_dir(cfg)
    experiment: Dict[str, Any] = {}
    for task in cfg.tasks:
        task_package = _load_or_build_task_package(raw_data, task, cfg, cache_dir, force_rebuild, progress)
        obj: Dict[str, Any] = {"task_package": task_package}
        if need_outer:
            obj["outer_train_samples"] = _load_or_build_sample_cache(task_package, "outer_train_samples", task_package["holdout_split"]["train_origins_for_valid"], cfg, cache_dir, force_rebuild, progress)
            obj["outer_valid_samples"] = _load_or_build_sample_cache(task_package, "outer_valid_samples", task_package["holdout_split"]["valid_origins"], cfg, cache_dir, force_rebuild, progress)
        if need_holdout:
            obj["final_train_samples"] = _load_or_build_sample_cache(task_package, "final_train_samples", task_package["strict_inner_split"]["final_train_origins"], cfg, cache_dir, force_rebuild, progress)
            obj["internal_valid_samples"] = _load_or_build_sample_cache(task_package, "internal_valid_samples", task_package["strict_inner_split"]["internal_valid_origins"], cfg, cache_dir, force_rebuild, progress)
            obj["test_samples"] = _load_or_build_sample_cache(task_package, "test_samples", task_package["holdout_split"]["test_origins"], cfg, cache_dir, force_rebuild, progress)
        if need_rolling:
            rolls = []
            for fold_id, fold in enumerate(task_package["rolling_folds"], start=1):
                train_samples = _load_or_build_sample_cache(task_package, f"rolling_fold_{fold_id}_train_samples", fold["train_origins"], cfg, cache_dir, force_rebuild, progress)
                valid_samples = _load_or_build_sample_cache(task_package, f"rolling_fold_{fold_id}_valid_samples", fold["valid_origins"], cfg, cache_dir, force_rebuild, progress)
                rolls.append({"fold_id": fold_id, "train_samples": train_samples, "valid_samples": valid_samples, "train_origins": fold["train_origins"], "valid_origins": fold["valid_origins"]})
            obj["rolling_fold_samples"] = rolls
        experiment[task] = obj
    return experiment


def print_experiment_summary(experiment: Dict[str, Any]) -> None:
    print("=" * 100)
    print("M5 EXPERIMENT SUMMARY")
    print("=" * 100)
    for task, obj in experiment.items():
        pkg = obj["task_package"]
        print(f"\nTask: {task}")
        print(f"  # aggregate units: {len(pkg['agg_ids'])}")
        print(f"  # child sku ids:    {sum(v['y_full'].shape[0] for v in pkg['aggregate_arrays'].values())}")
        print(f"  # time points:      {len(pkg['all_times'])}")
        print(f"  sampled store-SKUs: frac={pkg.get('store_sku_sample_frac', 1.0)} seed={pkg.get('store_sku_sample_seed', 42)}")
        print(f"  hist features:      {pkg['hist_feature_cols']}")
        print(f"  future features:    {pkg['future_feature_cols']}")
        if "outer_train_samples" in obj:
            print(f"  outer train samples: {len(obj['outer_train_samples'])}")
        if "outer_valid_samples" in obj:
            print(f"  outer valid samples: {len(obj['outer_valid_samples'])}")
        if "final_train_samples" in obj:
            print(f"  final train samples: {len(obj['final_train_samples'])}")
        if "internal_valid_samples" in obj:
            print(f"  internal valid samples: {len(obj['internal_valid_samples'])}")
        if "test_samples" in obj:
            print(f"  test samples: {len(obj['test_samples'])}")
        if "rolling_fold_samples" in obj:
            print(f"  rolling folds: {len(obj['rolling_fold_samples'])}")


# Public lightweight wrappers for task-wise lazy loading
def get_mode_split_specs(task_package: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    if mode == "holdout":
        return [{
            "fold_id": 0,
            "train_split_name": "final_train_samples",
            "train_origins": task_package["strict_inner_split"]["final_train_origins"],
            "valid_split_name": "internal_valid_samples",
            "valid_origins": task_package["strict_inner_split"]["internal_valid_origins"],
            "test_split_name": "test_samples",
            "test_origins": task_package["holdout_split"]["test_origins"],
        }]
    if mode == "rolling":
        specs: List[Dict[str, Any]] = []
        for fold_id, fold in enumerate(task_package["rolling_folds"], start=1):
            specs.append({
                "fold_id": fold_id,
                "train_split_name": f"rolling_fold_{fold_id}_train_samples",
                "train_origins": fold["train_origins"],
                "valid_split_name": f"rolling_fold_{fold_id}_valid_samples",
                "valid_origins": fold["valid_origins"],
            })
        return specs
    raise ValueError(f"Unsupported mode: {mode}")


def load_task_split_samples(
    task_package: Dict[str, Any],
    split_name: str,
    origin_indices: Sequence[int],
    cfg: M5ExperimentConfig,
    cache_dir: Optional[str] = None,
    force_rebuild: bool = False,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    return _load_or_build_sample_cache(task_package, split_name, origin_indices, cfg, cache_dir, force_rebuild, progress)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/cache M5 experiment artifacts and print a summary.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="store_dept,store_cat,state_dept")
    parser.add_argument("--t-hist", type=int, default=364)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--valid-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=4)
    parser.add_argument("--internal-valid-size", type=int, default=3)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--min-train-origins", type=int, default=20)
    parser.add_argument("--min-final-train-origins", type=int, default=20)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--sku-universe-mode", type=str, default="history", choices=["history", "history_or_future"])
    parser.add_argument("--store-sku-sample-frac", type=float, default=1.0, help="Fraction of unique item-store units to keep globally across all tasks.")
    parser.add_argument("--store-sku-sample-seed", type=int, default=42, help="Random seed for reproducible global item-store sampling.")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--outer", action="store_true")
    args = parser.parse_args()

    cfg = M5ExperimentConfig(
        data_dir=args.data_dir,
        t_hist=args.t_hist,
        horizon=args.horizon,
        tasks=tuple([t.strip() for t in args.tasks.split(",") if t.strip()]),
        valid_size=args.valid_size,
        test_size=args.test_size,
        internal_valid_size=args.internal_valid_size,
        gap=args.gap,
        min_train_origins=args.min_train_origins,
        min_final_train_origins=args.min_final_train_origins,
        fill_value=args.fill_value,
        sku_universe_mode=args.sku_universe_mode,
        store_sku_sample_frac=args.store_sku_sample_frac,
        store_sku_sample_seed=args.store_sku_sample_seed,
    )
    exp = build_m5_experiment_samples(
        raw_data=None,
        cfg=cfg,
        need_outer=args.outer,
        need_holdout=(args.holdout or (not args.rolling and not args.outer)),
        need_rolling=args.rolling,
        cache_dir=(args.cache_dir or None),
        force_rebuild=args.force_rebuild_cache,
        progress=(not args.quiet_progress),
    )
    print_experiment_summary(exp)
