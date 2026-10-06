from __future__ import annotations

import argparse
import gc
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prepare_m5_experiments import (
    M5ExperimentConfig,
    build_task_package,
    default_cache_dir,
    get_mode_split_specs,
    load_task_split_samples,
)
from compute_m5_metrics import (
    build_task_aggregate_daily,
    prepare_aggregate_history_store,
    compute_wrmsse_from_samples,
    compute_wspl_from_samples,
    summarize_point_results,
    summarize_quantile_results,
    build_main_point_table,
    build_main_quantile_table,
)

from experiment_tracking import (
    WandbRunWrapper,
    parse_tags,
    save_json,
)


# ============================================================
# 0) Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def configure_torch_runtime(device: torch.device, cpu_threads: int = 1) -> None:
    """Keep CPU training stable across environments by limiting intra/inter-op threads."""
    if device.type != "cpu":
        return
    cpu_threads = max(1, int(cpu_threads))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(cpu_threads)
    except RuntimeError:
        # Can be raised if inter-op threads were already initialized earlier.
        pass

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# ============================================================
# 1) Scalers
# ============================================================

class FeatureStandardScaler:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, arrays: List[np.ndarray]) -> "FeatureStandardScaler":
        total_count = 0
        sum_x = None
        sum_x2 = None
        fdim = None
        for x in arrays:
            arr = np.asarray(x, dtype=np.float32)
            if arr.size == 0:
                continue
            if fdim is None:
                fdim = arr.shape[-1]
                sum_x = np.zeros((fdim,), dtype=np.float64)
                sum_x2 = np.zeros((fdim,), dtype=np.float64)
            flat = arr.reshape(-1, fdim).astype(np.float64, copy=False)
            sum_x += flat.sum(axis=0)
            sum_x2 += np.square(flat).sum(axis=0)
            total_count += flat.shape[0]
        if total_count == 0:
            raise ValueError("Cannot fit FeatureStandardScaler on empty arrays.")
        mean = sum_x / float(total_count)
        var = np.maximum(sum_x2 / float(total_count) - np.square(mean), self.eps ** 2)
        self.mean_ = mean.reshape(1, -1).astype(np.float32)
        self.std_ = np.sqrt(var).reshape(1, -1).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean_) / self.std_).astype(np.float32, copy=False)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * self.std_ + self.mean_).astype(np.float32, copy=False)


class ArrayStandardScaler:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, arrays: List[np.ndarray]) -> "ArrayStandardScaler":
        total_count = 0
        sum_x = 0.0
        sum_x2 = 0.0
        for x in arrays:
            arr = np.asarray(x, dtype=np.float32)
            if arr.size == 0:
                continue
            flat = arr.reshape(-1).astype(np.float64, copy=False)
            sum_x += float(flat.sum())
            sum_x2 += float(np.square(flat).sum())
            total_count += flat.size
        if total_count == 0:
            raise ValueError("Cannot fit ArrayStandardScaler on empty arrays.")
        mean = sum_x / float(total_count)
        var = max(sum_x2 / float(total_count) - mean * mean, self.eps ** 2)
        self.mean_ = np.asarray([[mean]], dtype=np.float32)
        self.std_ = np.asarray([[np.sqrt(var)]], dtype=np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean_.squeeze()) / self.std_.squeeze()).astype(np.float32, copy=False)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return (arr * self.std_.squeeze() + self.mean_.squeeze()).astype(np.float32, copy=False)


def fit_sample_scalers(train_samples: List[Dict[str, Any]], use_static: bool = False) -> Dict[str, Any]:
    scalers: Dict[str, Any] = {}
    scalers["y_hist"] = ArrayStandardScaler().fit([s["y_hist"] for s in train_samples])
    scalers["y_target"] = ArrayStandardScaler().fit([s["y_target"] for s in train_samples])
    scalers["x_hist"] = FeatureStandardScaler().fit([s["x_hist"] for s in train_samples])
    scalers["x_future"] = FeatureStandardScaler().fit([s["x_future"] for s in train_samples])
    if use_static:
        scalers["static"] = FeatureStandardScaler().fit([s["static_feat"] for s in train_samples])
    return scalers


def transform_single_sample(sample: Dict[str, Any], scalers: Dict[str, Any], use_static: bool = False) -> Dict[str, np.ndarray]:
    item = {
        "y_hist": scalers["y_hist"].transform(sample["y_hist"]),
        "y_target": scalers["y_target"].transform(sample["y_target"]),
        "x_hist": scalers["x_hist"].transform(sample["x_hist"]),
        "x_future": scalers["x_future"].transform(sample["x_future"]),
    }
    if use_static and "static_feat" in sample:
        item["static_feat"] = scalers["static"].transform(sample["static_feat"])
    return item


def materialize_transformed_samples(samples: List[Dict[str, Any]], scalers: Dict[str, Any], use_static: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sample in samples:
        item = transform_single_sample(sample, scalers, use_static=use_static)
        if "meta" in sample:
            item["meta"] = sample["meta"]
        out.append(item)
    return out


def compute_raw_wape(preds: np.ndarray, samples_raw: List[Dict[str, Any]]) -> float:
    abs_err = 0.0
    abs_denom = 0.0
    for i, s in enumerate(samples_raw):
        y = np.asarray(s["y_target"], dtype=np.float32)
        abs_err += float(np.abs(np.asarray(preds[i], dtype=np.float32) - y).sum())
        abs_denom += float(np.abs(y).sum())
    return abs_err / (abs_denom + 1e-8)


def annotate_series_rows(
    series_records: Sequence[Dict[str, Any]],
    task: str,
    model: str,
    base_model: str,
    fold: int,
    metric_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in series_records:
        row = {
            "task": task,
            "fold": int(fold),
            "model": model,
            "base_model": base_model,
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


# ============================================================
# 2) Dataset / collate
# ============================================================

class RetailAggregateDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        scalers: Optional[Dict[str, Any]] = None,
        use_static: bool = False,
        materialize: bool = False,
    ):
        self.use_static = use_static
        self.materialize = bool(materialize and scalers is not None)
        if self.materialize:
            self.samples = materialize_transformed_samples(samples, scalers, use_static=use_static)
            self.scalers = None
        else:
            self.samples = samples
            self.scalers = scalers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        if self.scalers is not None:
            s = transform_single_sample(s, self.scalers, use_static=self.use_static)
        item = {
            "y_hist": torch.from_numpy(np.asarray(s["y_hist"], dtype=np.float32)),
            "x_hist": torch.from_numpy(np.asarray(s["x_hist"], dtype=np.float32)),
            "x_future": torch.from_numpy(np.asarray(s["x_future"], dtype=np.float32)),
            "y_target": torch.from_numpy(np.asarray(s["y_target"], dtype=np.float32)),
        }
        if self.use_static and "static_feat" in s:
            item["static_feat"] = torch.from_numpy(np.asarray(s["static_feat"], dtype=np.float32))
        return item


def retail_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    has_static = "static_feat" in batch[0]
    n_list = [item["y_hist"].shape[0] for item in batch]
    n_max = max(n_list)
    T = batch[0]["y_hist"].shape[1]
    H = batch[0]["x_future"].shape[1]
    F_hist = batch[0]["x_hist"].shape[2]
    F_future = batch[0]["x_future"].shape[2]

    y_hist = torch.zeros(B, n_max, T, 1, dtype=torch.float32)
    x_hist = torch.zeros(B, n_max, T, F_hist, dtype=torch.float32)
    x_future = torch.zeros(B, n_max, H, F_future, dtype=torch.float32)
    y_target = torch.stack([item["y_target"] for item in batch], dim=0)
    sku_mask = torch.zeros(B, n_max, dtype=torch.bool)

    static_feat = None
    if has_static:
        F_static = batch[0]["static_feat"].shape[1]
        static_feat = torch.zeros(B, n_max, F_static, dtype=torch.float32)

    for i, item in enumerate(batch):
        n_i = item["y_hist"].shape[0]
        y_hist[i, :n_i] = item["y_hist"]
        x_hist[i, :n_i] = item["x_hist"]
        x_future[i, :n_i] = item["x_future"]
        sku_mask[i, :n_i] = True
        if has_static:
            static_feat[i, :n_i] = item["static_feat"]

    out = {
        "y_hist": y_hist,
        "x_hist": x_hist,
        "x_future": x_future,
        "y_target": y_target,
        "sku_mask": sku_mask,
    }
    if has_static:
        out["static_feat"] = static_feat
    return out


# ============================================================
# 3) Models
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SkuFutureTemporalCNNEncoder(nn.Module):
    def __init__(
        self,
        future_feat_dim: int,
        hidden_dim: int,
        static_feat_dim: int = 0,
        dropout: float = 0.1,
        num_blocks: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_feat_dim = static_feat_dim
        if static_feat_dim > 0:
            self.static_proj = MLP(static_feat_dim, [hidden_dim], hidden_dim, dropout)
            static_emb_dim = hidden_dim
        else:
            self.static_proj = None
            static_emb_dim = 0
        self.input_proj = nn.Conv1d(future_feat_dim + static_emb_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_future: torch.Tensor, static_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_future: [B, N, H, F]
        B, N, H, F = x_future.shape
        if self.static_proj is not None and static_feat is not None:
            static_emb = self.static_proj(static_feat)
            static_seq = static_emb.unsqueeze(2).expand(B, N, H, static_emb.size(-1))
            x = torch.cat([x_future, static_seq], dim=-1)
        else:
            x = x_future
        x = x.reshape(B * N, H, -1).transpose(1, 2)  # [B*N, F, H]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # [B*N, H, D]
        x = self.out_norm(x)
        return x.reshape(B, N, H, self.hidden_dim)


class HorizonWiseGatedPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sku_horizon_embeddings: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
        # sku_horizon_embeddings: [B, N, H, D]
        value = self.value_proj(sku_horizon_embeddings)
        logits = self.gate(sku_horizon_embeddings).squeeze(-1)  # [B, N, H]
        valid = sku_mask.unsqueeze(-1).expand_as(logits)
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        pooled = (weights.unsqueeze(-1) * value).sum(dim=1)  # [B, H, D]
        active_count = sku_mask.sum(dim=1, keepdim=True).float().unsqueeze(1).expand(-1, pooled.size(1), -1)
        return self.out_proj(torch.cat([pooled, active_count], dim=-1))


def masked_mean_over_skus(x: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
    weights = sku_mask.float().unsqueeze(-1).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


def masked_std_over_skus(x: torch.Tensor, sku_mask: torch.Tensor, mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    weights = sku_mask.float().unsqueeze(-1).unsqueeze(-1)
    if mean is None:
        mean = masked_mean_over_skus(x, sku_mask)
    denom = weights.sum(dim=1).clamp_min(1.0)
    var = (((x - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / denom
    return torch.sqrt(var.clamp_min(1e-6))


def masked_max_over_skus(x: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
    mask = sku_mask.unsqueeze(-1).unsqueeze(-1)
    masked = x.masked_fill(~mask, torch.finfo(x.dtype).min)
    max_vals = masked.max(dim=1).values
    no_valid = (~sku_mask).all(dim=1, keepdim=True).unsqueeze(-1)
    return torch.where(no_valid, torch.zeros_like(max_vals), max_vals)


class DeepSetHorizonPooling(nn.Module):
    """Permutation-invariant set encoder without explicit SKU-SKU interactions."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sku_horizon_embeddings: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
        phi_x = self.phi(sku_horizon_embeddings)
        mean = masked_mean_over_skus(phi_x, sku_mask)
        std = masked_std_over_skus(phi_x, sku_mask, mean=mean)
        max_vals = masked_max_over_skus(phi_x, sku_mask)
        active_count = sku_mask.sum(dim=1, keepdim=True).float().unsqueeze(1).expand(-1, mean.size(1), -1)
        pooled = torch.cat([mean, std, max_vals, active_count], dim=-1)
        return self.rho(pooled)


class SetAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_seeds = max(1, int(num_seeds))
        self.seed_vectors = nn.Parameter(torch.randn(self.num_seeds, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        seeds = self.seed_vectors.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attn(seeds, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = self.norm1(seeds + self.dropout(pooled))
        ff_out = self.ffn(pooled)
        pooled = self.norm2(pooled + self.dropout(ff_out))
        return pooled


class MultiheadCrossAttentionBlock(nn.Module):
    """Residual cross-attention block used by ISAB/PMA-style pooling."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.norm1(query + self.dropout(attn_out))
        ff_out = self.ffn(out)
        return self.norm2(out + self.dropout(ff_out))


class InducedSetAttentionBlock(nn.Module):
    """
    Memory-safe alternative to SAB.

    Complexity is O(B * H * N * M) instead of O(B * H * N^2), where M is the number
    of inducing points. This keeps SetTransformer usable for large SKU sets (e.g. 800+).
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_inducing_points: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_inducing_points = max(1, int(num_inducing_points))
        self.inducing_points = nn.Parameter(torch.randn(self.num_inducing_points, hidden_dim) * 0.02)
        self.induce = MultiheadCrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.project_back = MultiheadCrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        inducing = self.inducing_points.unsqueeze(0).expand(B, -1, -1)
        induced = self.induce(inducing, x, key_padding_mask=key_padding_mask)
        out = self.project_back(x, induced, key_padding_mask=None)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class SetTransformerHorizonPooling(nn.Module):
    """Attention-based set encoder with memory-safe induced attention per horizon."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        num_seeds: int = 1,
        dropout: float = 0.1,
        num_inducing_points: int = 32,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads} for SetTransformer.")
        self.hidden_dim = hidden_dim
        self.num_seeds = max(1, int(num_seeds))
        self.num_inducing_points = max(1, int(num_inducing_points))
        self.sab_layers = nn.ModuleList([
            InducedSetAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_inducing_points=self.num_inducing_points,
                dropout=dropout,
            )
            for _ in range(max(1, int(num_sab_layers)))
        ])
        self.pma = PoolingByMultiheadAttention(hidden_dim=hidden_dim, num_heads=num_heads, num_seeds=self.num_seeds, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_seeds + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sku_horizon_embeddings: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
        # sku_horizon_embeddings: [B, N, H, D]
        B, N, H, D = sku_horizon_embeddings.shape
        x = sku_horizon_embeddings.permute(0, 2, 1, 3).reshape(B * H, N, D)
        invalid = (~sku_mask).unsqueeze(1).expand(B, H, N).reshape(B * H, N)
        for sab in self.sab_layers:
            x = sab(x, key_padding_mask=invalid)
        pooled = self.pma(x, key_padding_mask=invalid).reshape(B, H, self.num_seeds * D)
        active_count = sku_mask.sum(dim=1, keepdim=True).float().unsqueeze(1).expand(-1, H, -1)
        return self.out_proj(torch.cat([pooled, active_count], dim=-1))


class AggregateFutureSummaryEncoder(nn.Module):
    def __init__(self, future_feat_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = MLP(future_feat_dim * 2 + 1, [hidden_dim, hidden_dim], hidden_dim, dropout)

    def forward(self, x_future: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
        # x_future: [B, N, H, F]
        weights = sku_mask.float().unsqueeze(-1).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        mean = (x_future * weights).sum(dim=1) / denom
        centered = (x_future - mean.unsqueeze(1)) * weights
        var = (centered ** 2).sum(dim=1) / denom
        std = torch.sqrt(var.clamp_min(1e-6))
        active = sku_mask.sum(dim=1, keepdim=True).float().unsqueeze(1).expand(-1, mean.size(1), -1)
        return self.input_proj(torch.cat([mean, std, active], dim=-1))


class AggregateHistoryBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        recent_window: int = 14,
        lag_windows: Sequence[int] = (7, 14, 28, 56),
        stat_windows: Sequence[int] = (7, 28, 56),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.recent_window = int(recent_window)
        self.lag_windows = tuple(int(x) for x in lag_windows)
        self.stat_windows = tuple(int(x) for x in stat_windows)
        self.hist_rnn = nn.GRU(1, hidden_dim, batch_first=True)
        stats_dim = len(self.lag_windows) + 2 * len(self.stat_windows)
        self.context_proj = MLP(hidden_dim + stats_dim + 1, [hidden_dim, hidden_dim], hidden_dim, dropout)

    def _lag_and_stat_features(self, values: torch.Tensor) -> torch.Tensor:
        feats = []
        T = values.size(-1)
        for lag in self.lag_windows:
            idx = max(T - lag, 0)
            feats.append(values[:, idx])
        for window in self.stat_windows:
            w = min(window, T)
            segment = values[:, -w:]
            feats.append(segment.mean(dim=-1))
            feats.append(segment.std(dim=-1, unbiased=False))
        return torch.stack(feats, dim=-1)

    def forward(self, y_hist: torch.Tensor, sku_mask: torch.Tensor) -> torch.Tensor:
        agg_hist = (y_hist.squeeze(-1) * sku_mask.unsqueeze(-1)).sum(dim=1)  # [B, T]
        recent = min(self.recent_window, agg_hist.size(1))
        recent_seq = agg_hist[:, -recent:].unsqueeze(-1)
        _, h_recent = self.hist_rnn(recent_seq)
        h_recent = h_recent[-1]
        hist_stats = self._lag_and_stat_features(agg_hist)
        active_count = sku_mask.sum(dim=1, keepdim=True).float()
        return self.context_proj(torch.cat([h_recent, hist_stats, active_count], dim=-1))


class HorizonForecastHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        quantiles: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        monotone_quantiles: bool = True,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.quantiles = list(quantiles) if quantiles is not None else None
        self.monotone_quantiles = monotone_quantiles
        self.horizon_embedding = nn.Embedding(self.horizon, hidden_dim)
        out_dim = 1 if self.quantiles is None else len(self.quantiles)
        self.head = MLP(hidden_dim * 3, [hidden_dim, hidden_dim], out_dim, dropout)

    def forward(self, pooled_future: torch.Tensor, aggregate_context: torch.Tensor) -> torch.Tensor:
        # pooled_future: [B, H, D]
        B, H, D = pooled_future.shape
        hist_ctx = aggregate_context.unsqueeze(1).expand(B, H, D)
        horizon_idx = torch.arange(H, device=pooled_future.device)
        horizon_emb = self.horizon_embedding(horizon_idx).unsqueeze(0).expand(B, H, D)
        out = self.head(torch.cat([pooled_future, hist_ctx, horizon_emb], dim=-1))
        if self.quantiles is None:
            return out.squeeze(-1)
        if self.monotone_quantiles:
            out, _ = torch.sort(out, dim=-1)
        return out


class AggregateHistOnlyForecaster(nn.Module):
    def __init__(
        self,
        hist_feat_dim: int,
        future_feat_dim: int,
        hidden_dim: int,
        horizon: int,
        recent_window: int = 14,
        lag_windows: Sequence[int] = (7, 14, 28, 56),
        stat_windows: Sequence[int] = (7, 28, 56),
        static_feat_dim: int = 0,
        quantiles: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        monotone_quantiles: bool = True,
    ):
        super().__init__()
        _ = hist_feat_dim, future_feat_dim, static_feat_dim
        self.hidden_dim = hidden_dim
        self.aggregate_context = AggregateHistoryBranch(
            hidden_dim=hidden_dim,
            recent_window=recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            dropout=dropout,
        )
        self.head = HorizonForecastHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            quantiles=quantiles,
            dropout=dropout,
            monotone_quantiles=monotone_quantiles,
        )

    def forward(self, y_hist, x_hist, x_future, sku_mask, static_feat=None):
        del x_hist, x_future, static_feat
        aggregate_context = self.aggregate_context(y_hist=y_hist, sku_mask=sku_mask)
        B = aggregate_context.size(0)
        pooled_future = torch.zeros(B, self.head.horizon, self.hidden_dim, device=aggregate_context.device, dtype=aggregate_context.dtype)
        return self.head(pooled_future=pooled_future, aggregate_context=aggregate_context)


class AggregateFutureSummaryForecaster(nn.Module):
    def __init__(
        self,
        hist_feat_dim: int,
        future_feat_dim: int,
        hidden_dim: int,
        horizon: int,
        recent_window: int = 14,
        lag_windows: Sequence[int] = (7, 14, 28, 56),
        stat_windows: Sequence[int] = (7, 28, 56),
        static_feat_dim: int = 0,
        quantiles: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        monotone_quantiles: bool = True,
    ):
        super().__init__()
        _ = hist_feat_dim, static_feat_dim
        self.future_summary = AggregateFutureSummaryEncoder(future_feat_dim=future_feat_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.aggregate_context = AggregateHistoryBranch(
            hidden_dim=hidden_dim,
            recent_window=recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            dropout=dropout,
        )
        self.head = HorizonForecastHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            quantiles=quantiles,
            dropout=dropout,
            monotone_quantiles=monotone_quantiles,
        )

    def forward(self, y_hist, x_hist, x_future, sku_mask, static_feat=None):
        del x_hist, static_feat
        pooled_future = self.future_summary(x_future=x_future, sku_mask=sku_mask)
        aggregate_context = self.aggregate_context(y_hist=y_hist, sku_mask=sku_mask)
        return self.head(pooled_future=pooled_future, aggregate_context=aggregate_context)


class CrossLevelTemporalCNNGatedForecaster(nn.Module):
    def __init__(
        self,
        hist_feat_dim: int,
        future_feat_dim: int,
        hidden_dim: int,
        horizon: int,
        recent_window: int = 14,
        lag_windows: Sequence[int] = (7, 14, 28, 56),
        stat_windows: Sequence[int] = (7, 28, 56),
        static_feat_dim: int = 0,
        quantiles: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        monotone_quantiles: bool = True,
        cnn_num_blocks: int = 3,
        cnn_kernel_size: int = 3,
    ):
        super().__init__()
        _ = hist_feat_dim  # kept for interface compatibility
        self.sku_encoder = SkuFutureTemporalCNNEncoder(
            future_feat_dim=future_feat_dim,
            hidden_dim=hidden_dim,
            static_feat_dim=static_feat_dim,
            dropout=dropout,
            num_blocks=cnn_num_blocks,
            kernel_size=cnn_kernel_size,
        )
        self.horizon_pool = HorizonWiseGatedPooling(hidden_dim=hidden_dim, dropout=dropout)
        self.aggregate_context = AggregateHistoryBranch(
            hidden_dim=hidden_dim,
            recent_window=recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            dropout=dropout,
        )
        self.head = HorizonForecastHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            quantiles=quantiles,
            dropout=dropout,
            monotone_quantiles=monotone_quantiles,
        )

    def forward(self, y_hist, x_hist, x_future, sku_mask, static_feat=None):
        del x_hist
        sku_future_states = self.sku_encoder(x_future=x_future, static_feat=static_feat)
        pooled_future = self.horizon_pool(sku_future_states, sku_mask)
        aggregate_context = self.aggregate_context(y_hist=y_hist, sku_mask=sku_mask)
        return self.head(pooled_future=pooled_future, aggregate_context=aggregate_context)


class CrossLevelDeepSetForecaster(nn.Module):
    def __init__(
        self,
        hist_feat_dim: int,
        future_feat_dim: int,
        hidden_dim: int,
        horizon: int,
        recent_window: int = 14,
        lag_windows: Sequence[int] = (7, 14, 28, 56),
        stat_windows: Sequence[int] = (7, 28, 56),
        static_feat_dim: int = 0,
        quantiles: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        monotone_quantiles: bool = True,
        cnn_num_blocks: int = 3,
        cnn_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()
        _ = hist_feat_dim, kwargs
        self.sku_encoder = SkuFutureTemporalCNNEncoder(
            future_feat_dim=future_feat_dim,
            hidden_dim=hidden_dim,
            static_feat_dim=static_feat_dim,
            dropout=dropout,
            num_blocks=cnn_num_blocks,
            kernel_size=cnn_kernel_size,
        )
        self.deepset_pool = DeepSetHorizonPooling(hidden_dim=hidden_dim, dropout=dropout)
        self.aggregate_context = AggregateHistoryBranch(
            hidden_dim=hidden_dim,
            recent_window=recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            dropout=dropout,
        )
        self.head = HorizonForecastHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            quantiles=quantiles,
            dropout=dropout,
            monotone_quantiles=monotone_quantiles,
        )

    def forward(self, y_hist, x_hist, x_future, sku_mask, static_feat=None):
        del x_hist
        sku_future_states = self.sku_encoder(x_future=x_future, static_feat=static_feat)
        pooled_future = self.deepset_pool(sku_future_states, sku_mask)
        aggregate_context = self.aggregate_context(y_hist=y_hist, sku_mask=sku_mask)
        return self.head(pooled_future=pooled_future, aggregate_context=aggregate_context)


class CrossLevelSetTransformerForecaster(nn.Module):
    def __init__(
        self,
        hist_feat_dim: int,
        future_feat_dim: int,
        hidden_dim: int,
        horizon: int,
        recent_window: int = 14,
        lag_windows: Sequence[int] = (7, 14, 28, 56),
        stat_windows: Sequence[int] = (7, 28, 56),
        static_feat_dim: int = 0,
        quantiles: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        monotone_quantiles: bool = True,
        cnn_num_blocks: int = 3,
        cnn_kernel_size: int = 3,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        num_seeds: int = 1,
        num_inducing_points: int = 32,
        **kwargs,
    ):
        super().__init__()
        _ = hist_feat_dim, kwargs
        self.sku_encoder = SkuFutureTemporalCNNEncoder(
            future_feat_dim=future_feat_dim,
            hidden_dim=hidden_dim,
            static_feat_dim=static_feat_dim,
            dropout=dropout,
            num_blocks=cnn_num_blocks,
            kernel_size=cnn_kernel_size,
        )
        self.set_transformer_pool = SetTransformerHorizonPooling(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_sab_layers=num_sab_layers,
            num_seeds=num_seeds,
            dropout=dropout,
            num_inducing_points=num_inducing_points,
        )
        self.aggregate_context = AggregateHistoryBranch(
            hidden_dim=hidden_dim,
            recent_window=recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            dropout=dropout,
        )
        self.head = HorizonForecastHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            quantiles=quantiles,
            dropout=dropout,
            monotone_quantiles=monotone_quantiles,
        )

    def forward(self, y_hist, x_hist, x_future, sku_mask, static_feat=None):
        del x_hist
        sku_future_states = self.sku_encoder(x_future=x_future, static_feat=static_feat)
        pooled_future = self.set_transformer_pool(sku_future_states, sku_mask)
        aggregate_context = self.aggregate_context(y_hist=y_hist, sku_mask=sku_mask)
        return self.head(pooled_future=pooled_future, aggregate_context=aggregate_context)

# ============================================================
# 4) Loss / regularization / training
# ============================================================

def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    q = torch.tensor(list(quantiles), device=pred_q.device, dtype=pred_q.dtype).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred_q
    return torch.maximum(q * err, (q - 1.0) * err).mean()


@dataclass
class FitConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_workers: int = 0
    max_grad_norm: float = 1.0
    patience: int = 5
    use_amp: bool = False
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    print_every: int = 1
    batch_log_interval: int = 0
    materialize_datasets: bool = False


@dataclass
class TrainRegularizationConfig:
    use_permutation_consistency: bool = True
    permutation_prob: float = 0.5
    consistency_weight: float = 0.01
    consistency_loss_type: str = "mse"
    permute_only_valid_skus: bool = True


def permute_sku_axis_batch(batch: Dict[str, torch.Tensor], permute_only_valid_skus: bool = True) -> Dict[str, torch.Tensor]:
    sku_mask = batch["sku_mask"]
    B, N = sku_mask.shape
    device = sku_mask.device
    perms = []
    for b in range(B):
        mask_b = sku_mask[b]
        valid_idx = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
        invalid_idx = torch.nonzero(~mask_b, as_tuple=False).squeeze(-1)
        if permute_only_valid_skus:
            shuffled_valid = valid_idx[torch.randperm(valid_idx.numel(), device=device)] if valid_idx.numel() > 0 else valid_idx
            perm_b = torch.cat([shuffled_valid, invalid_idx], dim=0)
        else:
            perm_b = torch.randperm(N, device=device)
        perms.append(perm_b)
    perms = torch.stack(perms, dim=0)

    def _perm(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[b, perms[b]] for b in range(B)], dim=0)

    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and k in {"y_hist", "x_hist", "x_future", "sku_mask", "static_feat"}:
            out[k] = _perm(v)
        else:
            out[k] = v
    return out


def compute_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Optional[Sequence[float]] = None) -> torch.Tensor:
    if quantiles is None:
        return nn.MSELoss()(pred, target)
    return pinball_loss(pred, target, quantiles)


def compute_total_loss(model: nn.Module, batch: Dict[str, torch.Tensor], quantiles: Optional[Sequence[float]], reg_cfg: Optional[TrainRegularizationConfig]) -> Dict[str, torch.Tensor]:
    pred = model(
        y_hist=batch["y_hist"],
        x_hist=batch["x_hist"],
        x_future=batch["x_future"],
        sku_mask=batch["sku_mask"],
        static_feat=batch.get("static_feat", None),
    )
    target = batch["y_target"]
    forecast_loss = compute_loss(pred, target, quantiles)
    total_loss = forecast_loss
    consistency_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

    if reg_cfg is not None and reg_cfg.use_permutation_consistency and random.random() < reg_cfg.permutation_prob:
        batch_perm = permute_sku_axis_batch(batch, reg_cfg.permute_only_valid_skus)
        pred_perm = model(
            y_hist=batch_perm["y_hist"],
            x_hist=batch_perm["x_hist"],
            x_future=batch_perm["x_future"],
            sku_mask=batch_perm["sku_mask"],
            static_feat=batch_perm.get("static_feat", None),
        )
        if reg_cfg.consistency_loss_type == "mse":
            consistency_loss = ((pred - pred_perm) ** 2).mean()
        else:
            consistency_loss = torch.abs(pred - pred_perm).mean()
        total_loss = forecast_loss + reg_cfg.consistency_weight * consistency_loss

    return {
        "pred": pred,
        "forecast_loss": forecast_loss,
        "consistency_loss": consistency_loss,
        "total_loss": total_loss,
    }


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    quantiles=None,
    reg_cfg=None,
    max_grad_norm=1.0,
    batch_log_interval: int = 0,
):
    model.train()
    n = 0
    sums = {"loss": 0.0, "forecast_loss": 0.0, "consistency_loss": 0.0}
    total_batches = len(loader)
    batch_log_interval = max(0, int(batch_log_interval))
    epoch_start = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = compute_total_loss(model, batch, quantiles, reg_cfg)
        out["total_loss"].backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bs = batch["y_target"].size(0)
        n += bs
        sums["loss"] += out["total_loss"].item() * bs
        sums["forecast_loss"] += out["forecast_loss"].item() * bs
        sums["consistency_loss"] += out["consistency_loss"].item() * bs

        if batch_log_interval > 0 and (batch_idx % batch_log_interval == 0 or batch_idx == total_batches):
            avg_loss = sums["loss"] / max(n, 1)
            elapsed = time.time() - epoch_start
            print(
                f"    batch {batch_idx:04d}/{total_batches:04d} | "
                f"avg_train_loss={avg_loss:.6f} | elapsed={elapsed:.1f}s",
                flush=True,
            )
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device, quantiles=None):
    model.eval()
    n = 0
    loss_sum = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = model(
            y_hist=batch["y_hist"],
            x_hist=batch["x_hist"],
            x_future=batch["x_future"],
            sku_mask=batch["sku_mask"],
            static_feat=batch.get("static_feat", None),
        )
        loss = compute_loss(pred, batch["y_target"], quantiles)
        bs = batch["y_target"].size(0)
        n += bs
        loss_sum += loss.item() * bs
    return {"loss": loss_sum / max(n, 1)}


@torch.no_grad()
def predict_dataset(model, dataset, device, batch_size=64, num_workers=0):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=retail_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    preds = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = model(
            y_hist=batch["y_hist"],
            x_hist=batch["x_hist"],
            x_future=batch["x_future"],
            sku_mask=batch["sku_mask"],
            static_feat=batch.get("static_feat", None),
        )
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def fit_model(
    model: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    device: torch.device,
    fit_cfg: FitConfig,
    quantiles: Optional[Sequence[float]] = None,
    reg_cfg: Optional[TrainRegularizationConfig] = None,
    epoch_logger: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(valid_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=fit_cfg.batch_size,
        shuffle=True,
        num_workers=fit_cfg.num_workers,
        collate_fn=retail_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=fit_cfg.batch_size,
        shuffle=False,
        num_workers=fit_cfg.num_workers,
        collate_fn=retail_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fit_cfg.lr, weight_decay=fit_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=fit_cfg.scheduler_factor, patience=fit_cfg.scheduler_patience)
    early = EarlyStopping(patience=fit_cfg.patience)

    history = {"train_loss": [], "valid_loss": []}
    best_state = None

    for epoch in range(1, fit_cfg.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            quantiles,
            reg_cfg,
            fit_cfg.max_grad_norm,
            fit_cfg.batch_log_interval,
        )
        valid_metrics = evaluate(model, valid_loader, device, quantiles)
        epoch_time = time.time() - epoch_start
        scheduler.step(valid_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_forecast_loss": float(train_metrics["forecast_loss"]),
            "train_consistency_loss": float(train_metrics["consistency_loss"]),
            "valid_loss": float(valid_metrics["loss"]),
            "lr": current_lr,
            "epoch_time_sec": float(epoch_time),
        }

        if epoch % fit_cfg.print_every == 0:
            print(
                f"[Epoch {epoch:03d}] train={train_metrics['loss']:.6f} "
                f"valid={valid_metrics['loss']:.6f} lr={current_lr:.6g} "
                f"time={epoch_time:.1f}s"
            )

        improved = early.step(valid_metrics["loss"])
        epoch_metrics["best_so_far"] = 1.0 if improved else 0.0
        epoch_metrics["best_valid_loss"] = float(early.best)

        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch_logger is not None:
            epoch_logger(epoch_metrics, epoch)
        if early.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ============================================================
# 5) Runners
# ============================================================

def normalize_model_name(model_name: str) -> str:
    return model_name[:-2] if model_name.endswith("_Q") else model_name


def build_model(
    model_name: str,
    hist_feat_dim: int,
    future_feat_dim: int,
    horizon: int,
    quantiles: Optional[Sequence[float]],
    hidden_dim: int,
    dropout: float,
    num_heads: int,
    num_sab_layers: int,
    num_seeds: int,
    num_inducing_points: int,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    cnn_num_blocks: int = 3,
    cnn_kernel_size: int = 3,
):
    base_model_name = normalize_model_name(model_name)
    common_kwargs = dict(
        hist_feat_dim=hist_feat_dim,
        future_feat_dim=future_feat_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        recent_window=recent_window,
        lag_windows=lag_windows,
        stat_windows=stat_windows,
        static_feat_dim=0,
        quantiles=quantiles,
        dropout=dropout,
        monotone_quantiles=monotone_quantiles,
    )
    temporal_cnn_kwargs = dict(
        **common_kwargs,
        cnn_num_blocks=cnn_num_blocks,
        cnn_kernel_size=cnn_kernel_size,
    )
    set_model_kwargs = dict(
        **temporal_cnn_kwargs,
        num_heads=num_heads,
        num_sab_layers=num_sab_layers,
        num_seeds=num_seeds,
        num_inducing_points=num_inducing_points,
    )
    if base_model_name in {"AggHistOnly", "AggregateHistOnly", "M0"}:
        return AggregateHistOnlyForecaster(**common_kwargs)
    if base_model_name in {"AggHistFutureSummary", "AggregateFutureSummary", "M1"}:
        return AggregateFutureSummaryForecaster(**common_kwargs)
    if base_model_name in {"SkuTemporalCNN", "TemporalCNNGated", "FullModel", "FullSkuTemporalCNN", "M3", "M3_FullSkuTemporalCNN"}:
        return CrossLevelTemporalCNNGatedForecaster(**temporal_cnn_kwargs)
    if base_model_name in {"DeepSets", "CrossLevelDeepSetForecaster"}:
        return CrossLevelDeepSetForecaster(**temporal_cnn_kwargs)
    if base_model_name in {"SetTransformer", "CrossLevelSetTransformerForecaster"}:
        return CrossLevelSetTransformerForecaster(**set_model_kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")


def run_one_setting(
    model_name: str,
    train_samples_raw: List[Dict[str, Any]],
    valid_samples_raw: List[Dict[str, Any]],
    test_like_samples_raw: List[Dict[str, Any]],
    agg_history_store: Dict[str, pd.DataFrame],
    task: str,
    device: torch.device,
    fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    quantiles: Optional[Sequence[float]],
    hidden_dim: int,
    dropout: float,
    num_heads: int,
    num_sab_layers: int,
    num_seeds: int,
    num_inducing_points: int,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    cnn_num_blocks: int = 3,
    cnn_kernel_size: int = 3,
    epoch_logger: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> Dict[str, Any]:
    scalers = fit_sample_scalers(train_samples_raw, use_static=False)

    train_ds = RetailAggregateDataset(train_samples_raw, scalers=scalers, use_static=False, materialize=fit_cfg.materialize_datasets)
    valid_ds = RetailAggregateDataset(valid_samples_raw, scalers=scalers, use_static=False, materialize=fit_cfg.materialize_datasets)
    test_ds = RetailAggregateDataset(test_like_samples_raw, scalers=scalers, use_static=False, materialize=fit_cfg.materialize_datasets)

    hist_feat_dim = train_samples_raw[0]["x_hist"].shape[-1]
    future_feat_dim = train_samples_raw[0]["x_future"].shape[-1]
    horizon = train_samples_raw[0]["y_target"].shape[0]

    model = build_model(
        model_name=model_name,
        hist_feat_dim=hist_feat_dim,
        future_feat_dim=future_feat_dim,
        horizon=horizon,
        quantiles=quantiles,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_heads=num_heads,
        num_sab_layers=num_sab_layers,
        num_seeds=num_seeds,
        num_inducing_points=num_inducing_points,
        recent_window=recent_window,
        lag_windows=lag_windows,
        stat_windows=stat_windows,
        monotone_quantiles=monotone_quantiles,
        cnn_num_blocks=cnn_num_blocks,
        cnn_kernel_size=cnn_kernel_size,
    )

    model, history = fit_model(model, train_ds, valid_ds, device, fit_cfg, quantiles, reg_cfg, epoch_logger=epoch_logger)
    preds_std = predict_dataset(model, test_ds, device=device, batch_size=fit_cfg.batch_size, num_workers=fit_cfg.num_workers)
    preds = scalers["y_target"].inverse_transform(preds_std)

    if quantiles is None:
        metric = compute_wrmsse_from_samples(
            samples_raw=test_like_samples_raw,
            preds=preds,
            agg_history_store=agg_history_store,
            task=task,
            horizon=horizon,
        )
        wape = compute_raw_wape(preds, test_like_samples_raw)
        out = {
            "task": task,
            "model": model_name,
            "base_model": normalize_model_name(model_name),
            "wrmsse": metric["wrmsse"],
            "wape": wape,
            "history": history,
            "series_rows": annotate_series_rows(
                metric.get("series_records", []),
                task=task,
                model=model_name,
                base_model=normalize_model_name(model_name),
                fold=-1,
                metric_name="rmsse",
            ),
        }
    else:
        metric = compute_wspl_from_samples(
            samples_raw=test_like_samples_raw,
            pred_quantiles=preds,
            quantiles=quantiles,
            agg_history_store=agg_history_store,
            task=task,
            horizon=horizon,
        )
        out = {
            "task": task,
            "model": model_name,
            "base_model": normalize_model_name(model_name),
            "wspl": metric["wspl"],
            "wape": np.nan,
            "history": history,
            "series_rows": annotate_series_rows(
                metric.get("series_records", []),
                task=task,
                model=model_name,
                base_model=normalize_model_name(model_name),
                fold=-1,
                metric_name="mean_spl",
            ),
        }
    return out


def run_task_mode(
    task_name: str,
    task_obj: Dict[str, Any],
    agg_history_store: Dict[str, pd.DataFrame],
    device: torch.device,
    fit_cfg: FitConfig,
    reg_cfg: TrainRegularizationConfig,
    quantiles: Optional[Sequence[float]],
    mode: str,
    hidden_dim: int,
    dropout: float,
    num_heads: int,
    num_sab_layers: int,
    num_seeds: int,
    num_inducing_points: int,
    recent_window: int,
    lag_windows: Sequence[int],
    stat_windows: Sequence[int],
    monotone_quantiles: bool,
    cnn_num_blocks: int = 3,
    cnn_kernel_size: int = 3,
    model_names: Optional[Sequence[str]] = None,
    wandb_run: Optional[WandbRunWrapper] = None,
    return_series_rows: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    series_rows: List[Dict[str, Any]] = []
    if model_names is None or len(model_names) == 0:
        model_names = ["DeepSets", "SetTransformer"]

    if mode == "rolling":
        for fold_obj in task_obj["rolling_fold_samples"]:
            fold_id = int(fold_obj["fold_id"])
            train_samples_raw = fold_obj["train_samples"]
            valid_samples_raw = fold_obj["valid_samples"]

            for model_name in model_names:
                print(f"Task={task_name} Fold={fold_id} Model={model_name}")
                epoch_logger = None
                if wandb_run is not None and wandb_run.active:
                    def epoch_logger(metrics: Dict[str, float], epoch: int, task_name=task_name, fold_id=fold_id, model_name=model_name):
                        wandb_run.log({
                            "train/task": task_name,
                            "train/fold": fold_id,
                            "train/model": model_name,
                            **{f"train/{k}": v for k, v in metrics.items()},
                        }, step=epoch)
                point_out = run_one_setting(
                    model_name=model_name,
                    train_samples_raw=train_samples_raw,
                    valid_samples_raw=valid_samples_raw,
                    test_like_samples_raw=valid_samples_raw,
                    agg_history_store=agg_history_store,
                    task=task_name,
                    device=device,
                    fit_cfg=fit_cfg,
                    reg_cfg=reg_cfg,
                    quantiles=None,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    num_sab_layers=num_sab_layers,
                    num_seeds=num_seeds,
                    num_inducing_points=num_inducing_points,
                    recent_window=recent_window,
                    lag_windows=lag_windows,
                    stat_windows=stat_windows,
                    monotone_quantiles=monotone_quantiles,
                    cnn_num_blocks=cnn_num_blocks,
                    cnn_kernel_size=cnn_kernel_size,
                    epoch_logger=epoch_logger,
                )
                point_out["fold"] = fold_id
                point_series_rows = point_out.pop("series_rows", [])
                for rec in point_series_rows:
                    rec["fold"] = fold_id
                series_rows.extend(point_series_rows)
                rows.append(point_out)
                if wandb_run is not None and wandb_run.active:
                    wandb_run.log({
                        "eval/task": task_name,
                        "eval/fold": fold_id,
                        "eval/model": point_out["model"],
                        "eval/wrmsse": float(point_out["wrmsse"]),
                        "eval/wape": float(point_out["wape"]),
                    })

                if quantiles is not None:
                    epoch_logger_q = None
                    if wandb_run is not None and wandb_run.active:
                        def epoch_logger_q(metrics: Dict[str, float], epoch: int, task_name=task_name, fold_id=fold_id, model_name=model_name):
                            wandb_run.log({
                                "train_q/task": task_name,
                                "train_q/fold": fold_id,
                                "train_q/model": model_name + "_Q",
                                **{f"train_q/{k}": v for k, v in metrics.items()},
                            }, step=epoch)
                    q_out = run_one_setting(
                        model_name=model_name + "_Q",
                        train_samples_raw=train_samples_raw,
                        valid_samples_raw=valid_samples_raw,
                        test_like_samples_raw=valid_samples_raw,
                        agg_history_store=agg_history_store,
                        task=task_name,
                        device=device,
                        fit_cfg=fit_cfg,
                        reg_cfg=reg_cfg,
                        quantiles=quantiles,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        num_heads=num_heads,
                        num_sab_layers=num_sab_layers,
                        num_seeds=num_seeds,
                        num_inducing_points=num_inducing_points,
                        recent_window=recent_window,
                        lag_windows=lag_windows,
                        stat_windows=stat_windows,
                        monotone_quantiles=monotone_quantiles,
                        cnn_num_blocks=cnn_num_blocks,
                        cnn_kernel_size=cnn_kernel_size,
                        epoch_logger=epoch_logger_q,
                    )
                    q_out["fold"] = fold_id
                    q_out["model"] = model_name + "_Q"
                    q_series_rows = q_out.pop("series_rows", [])
                    for rec in q_series_rows:
                        rec["fold"] = fold_id
                        rec["model"] = model_name + "_Q"
                        rec["base_model"] = model_name
                    series_rows.extend(q_series_rows)
                    rows.append(q_out)
                    if wandb_run is not None and wandb_run.active:
                        wandb_run.log({
                            "eval_q/task": task_name,
                            "eval_q/fold": fold_id,
                            "eval_q/model": q_out["model"],
                            "eval_q/wspl": float(q_out["wspl"]),
                        })

    elif mode == "holdout":
        train_samples_raw = task_obj["final_train_samples"]
        valid_samples_raw = task_obj["internal_valid_samples"]
        test_samples_raw = task_obj["test_samples"]

        for model_name in model_names:
            print(f"Task={task_name} Mode=holdout Model={model_name}")
            epoch_logger = None
            if wandb_run is not None and wandb_run.active:
                def epoch_logger(metrics: Dict[str, float], epoch: int, task_name=task_name, model_name=model_name):
                    wandb_run.log({
                        "train/task": task_name,
                        "train/fold": 0,
                        "train/model": model_name,
                        **{f"train/{k}": v for k, v in metrics.items()},
                    }, step=epoch)
            point_out = run_one_setting(
                model_name=model_name,
                train_samples_raw=train_samples_raw,
                valid_samples_raw=valid_samples_raw,
                test_like_samples_raw=test_samples_raw,
                agg_history_store=agg_history_store,
                task=task_name,
                device=device,
                fit_cfg=fit_cfg,
                reg_cfg=reg_cfg,
                quantiles=None,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_heads=num_heads,
                num_sab_layers=num_sab_layers,
                num_seeds=num_seeds,
                num_inducing_points=num_inducing_points,
                recent_window=recent_window,
                lag_windows=lag_windows,
                stat_windows=stat_windows,
                monotone_quantiles=monotone_quantiles,
                cnn_num_blocks=cnn_num_blocks,
                cnn_kernel_size=cnn_kernel_size,
                epoch_logger=epoch_logger,
            )
            point_out["fold"] = 0
            point_series_rows = point_out.pop("series_rows", [])
            for rec in point_series_rows:
                rec["fold"] = 0
            series_rows.extend(point_series_rows)
            rows.append(point_out)
            if wandb_run is not None and wandb_run.active:
                wandb_run.log({
                    "eval/task": task_name,
                    "eval/fold": 0,
                    "eval/model": point_out["model"],
                    "eval/wrmsse": float(point_out["wrmsse"]),
                    "eval/wape": float(point_out["wape"]),
                })

            if quantiles is not None:
                epoch_logger_q = None
                if wandb_run is not None and wandb_run.active:
                    def epoch_logger_q(metrics: Dict[str, float], epoch: int, task_name=task_name, model_name=model_name):
                        wandb_run.log({
                            "train_q/task": task_name,
                            "train_q/fold": 0,
                            "train_q/model": model_name + "_Q",
                            **{f"train_q/{k}": v for k, v in metrics.items()},
                        }, step=epoch)
                q_out = run_one_setting(
                    model_name=model_name + "_Q",
                    train_samples_raw=train_samples_raw,
                    valid_samples_raw=valid_samples_raw,
                    test_like_samples_raw=test_samples_raw,
                    agg_history_store=agg_history_store,
                    task=task_name,
                    device=device,
                    fit_cfg=fit_cfg,
                    reg_cfg=reg_cfg,
                    quantiles=quantiles,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    num_sab_layers=num_sab_layers,
                    num_seeds=num_seeds,
                    num_inducing_points=num_inducing_points,
                    recent_window=recent_window,
                    lag_windows=lag_windows,
                    stat_windows=stat_windows,
                    monotone_quantiles=monotone_quantiles,
                    cnn_num_blocks=cnn_num_blocks,
                    cnn_kernel_size=cnn_kernel_size,
                    epoch_logger=epoch_logger_q,
                )
                q_out["fold"] = 0
                q_out["model"] = model_name + "_Q"
                q_series_rows = q_out.pop("series_rows", [])
                for rec in q_series_rows:
                    rec["fold"] = 0
                    rec["model"] = model_name + "_Q"
                    rec["base_model"] = model_name
                series_rows.extend(q_series_rows)
                rows.append(q_out)
                if wandb_run is not None and wandb_run.active:
                    wandb_run.log({
                        "eval_q/task": task_name,
                        "eval_q/fold": 0,
                        "eval_q/model": q_out["model"],
                        "eval_q/wspl": float(q_out["wspl"]),
                    })
    else:
        raise ValueError("mode must be one of {'rolling','holdout'}")

    rows_df = pd.DataFrame(rows)
    series_df = pd.DataFrame(series_rows)
    if return_series_rows:
        return rows_df, series_df
    return rows_df


# ============================================================
# 6) CLI helpers
# ============================================================

def load_params_json(path: str) -> Dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("params JSON must be a dictionary")
    return data


def apply_overrides_from_params(args: argparse.Namespace, fit_cfg: FitConfig, reg_cfg: TrainRegularizationConfig) -> None:
    if not getattr(args, "params_json", ""):
        return
    params = load_params_json(args.params_json)
    args.hidden_dim = int(params.get("hidden_dim", args.hidden_dim))
    args.dropout = float(params.get("dropout", args.dropout))
    args.num_heads = int(params.get("num_heads", args.num_heads))
    args.num_sab_layers = int(params.get("num_sab_layers", args.num_sab_layers))
    args.num_seeds = int(params.get("num_seeds", args.num_seeds))
    args.recent_window = int(params.get("recent_window", args.recent_window))
    args.cnn_num_blocks = int(params.get("cnn_num_blocks", getattr(args, "cnn_num_blocks", 3)))
    args.cnn_kernel_size = int(params.get("cnn_kernel_size", getattr(args, "cnn_kernel_size", 3)))
    if "lag_windows" in params:
        args.lag_windows = ",".join(str(int(x)) for x in params["lag_windows"])
    if "stat_windows" in params:
        args.stat_windows = ",".join(str(int(x)) for x in params["stat_windows"])
    fit_cfg.lr = float(params.get("lr", fit_cfg.lr))
    fit_cfg.weight_decay = float(params.get("weight_decay", fit_cfg.weight_decay))
    fit_cfg.batch_size = int(params.get("batch_size", fit_cfg.batch_size))
    fit_cfg.epochs = int(params.get("epochs", fit_cfg.epochs))
    fit_cfg.patience = int(params.get("patience", fit_cfg.patience))
    reg_cfg.permutation_prob = float(params.get("permutation_prob", reg_cfg.permutation_prob))
    reg_cfg.consistency_weight = float(params.get("consistency_weight", reg_cfg.consistency_weight))


# ============================================================
# 7) Main script
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run proposed temporal-CNN aggregate forecasting models on M5 tasks.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./m5_neural_outputs")
    parser.add_argument("--mode", type=str, default="rolling", choices=["rolling", "holdout"])
    parser.add_argument("--tasks", type=str, default="store_dept,store_cat,state_dept")
    parser.add_argument("--quantiles", type=str, default="0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995")
    parser.add_argument("--model-names", type=str, default="DeepSets,SetTransformer", help="Comma-separated model names/aliases to run. Examples: M3_FullSkuTemporalCNN or DeepSets,SetTransformer.")
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-num-blocks", type=int, default=3, help="Number of temporal residual blocks in the SKU future encoder.")
    parser.add_argument("--cnn-kernel-size", type=int, default=3, choices=[3,5,7], help="Kernel size in the SKU future temporal CNN.")
    parser.add_argument("--recent-window", type=int, default=14, help="Recent aggregate-history length for the history branch.")
    parser.add_argument("--lag-windows", type=str, default="7,14,28,56", help="Comma-separated lag features extracted from history.")
    parser.add_argument("--stat-windows", type=str, default="7,28,56", help="Comma-separated rolling windows for mean/std features.")
    parser.add_argument("--disable-monotone-quantiles", action="store_true", help="Disable sorting-based monotone quantile output.")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-sab-layers", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-inducing-points", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-log-interval", type=int, default=0, help="Print intra-epoch batch progress every N batches. 0 disables batch-level logging.")
    parser.add_argument("--materialize-datasets", action="store_true", help="Pre-transform datasets once before training for easier debugging at the cost of extra memory.")
    parser.add_argument("--perm-prob", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.01)
    parser.add_argument("--use-permreg", action="store_true", help="Enable permutation consistency regularization. Disabled by default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-csv", type=str, default="", help="Optional CSV from run_m5_full_benchmark.py to merge with proposed-model rows.")
    parser.add_argument("--params-json", type=str, default="", help="Optional JSON file of tuned hyperparameters to override the CLI defaults.")
    parser.add_argument("--save-run-config", action="store_true", help="Save the effective run configuration as JSON in the output directory.")
    parser.add_argument("--use-wandb", action="store_true", help="Log training and evaluation metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="m5-cross-level-forecasting")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="m5,proposed-models")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--cache-dir", type=str, default="", help="Directory for reusable preprocessing caches.")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU intra/inter-op thread limit for stable CPU training.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild cached preprocessing artifacts.")
    parser.add_argument("--quiet-progress", action="store_true", help="Reduce preprocessing progress/ETA logging.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device, cpu_threads=args.cpu_threads)

    tasks = tuple([t.strip() for t in args.tasks.split(",") if t.strip()])
    quantiles = [float(x) for x in args.quantiles.split(",")] if args.quantiles else None
    model_names = [x.strip() for x in args.model_names.split(",") if x.strip()]
    lag_windows = tuple(int(x) for x in args.lag_windows.split(",") if x.strip())
    stat_windows = tuple(int(x) for x in args.stat_windows.split(",") if x.strip())
    monotone_quantiles = not args.disable_monotone_quantiles

    cfg = M5ExperimentConfig(
        data_dir=args.data_dir,
        t_hist=args.t_hist,
        horizon=args.horizon,
        tasks=tasks,
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

    fit_cfg = FitConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        batch_log_interval=args.batch_log_interval,
        materialize_datasets=args.materialize_datasets,
    )
    reg_cfg = TrainRegularizationConfig(
        use_permutation_consistency=bool(getattr(args, "use_permreg", False)),
        permutation_prob=args.perm_prob,
        consistency_weight=args.consistency_weight,
    )
    apply_overrides_from_params(args, fit_cfg, reg_cfg)

    cache_dir = args.cache_dir or default_cache_dir(cfg)
    print(f"Using cache directory: {cache_dir}")
    print("Building/loading cached task artifacts lazily...")

    os.makedirs(args.output_dir, exist_ok=True)
    effective_run_config = {
        "cfg": asdict(cfg),
        "fit_cfg": asdict(fit_cfg),
        "reg_cfg": asdict(reg_cfg),
        "device": str(device),
        "mode": args.mode,
        "tasks": list(cfg.tasks),
        "quantiles": quantiles,
        "cache_dir": cache_dir,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "cnn_num_blocks": args.cnn_num_blocks,
        "cnn_kernel_size": args.cnn_kernel_size,
        "num_heads": args.num_heads,
        "num_sab_layers": args.num_sab_layers,
        "num_seeds": args.num_seeds,
        "recent_window": args.recent_window,
        "lag_windows": list(lag_windows),
        "stat_windows": list(stat_windows),
        "monotone_quantiles": monotone_quantiles,
        "params_json": args.params_json,
        "benchmark_csv": args.benchmark_csv,
        "seed": args.seed,
    }
    if args.save_run_config:
        save_json(effective_run_config, os.path.join(args.output_dir, f"run_config_{args.mode}.json"))

    wandb_run = WandbRunWrapper(
        enabled=(args.use_wandb and args.wandb_mode != "disabled"),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        config=effective_run_config,
        mode=args.wandb_mode,
        job_type="proposed_models",
    )

    all_rows = []
    all_series_rows = []
    for task in cfg.tasks:
        print(f"\n=== Proposed models on task: {task} ===")
        task_package = build_task_package(
            raw_data=None,
            task=task,
            cfg=cfg,
            cache_dir=cache_dir,
            force_rebuild=args.force_rebuild_cache,
            progress=(not args.quiet_progress),
        )
        agg_daily = task_package["agg_daily_df"]
        hist_store = prepare_aggregate_history_store(agg_daily)

        if args.mode == "holdout":
            spec = get_mode_split_specs(task_package, "holdout")[0]
            task_obj = {
                "task_package": task_package,
                "final_train_samples": load_task_split_samples(task_package, spec["train_split_name"], spec["train_origins"], cfg, cache_dir=cache_dir, force_rebuild=args.force_rebuild_cache, progress=(not args.quiet_progress)),
                "internal_valid_samples": load_task_split_samples(task_package, spec["valid_split_name"], spec["valid_origins"], cfg, cache_dir=cache_dir, force_rebuild=args.force_rebuild_cache, progress=(not args.quiet_progress)),
                "test_samples": load_task_split_samples(task_package, spec["test_split_name"], spec["test_origins"], cfg, cache_dir=cache_dir, force_rebuild=args.force_rebuild_cache, progress=(not args.quiet_progress)),
            }
        else:
            rolling_fold_samples = []
            for spec in get_mode_split_specs(task_package, "rolling"):
                rolling_fold_samples.append({
                    "fold_id": int(spec["fold_id"]),
                    "train_samples": load_task_split_samples(task_package, spec["train_split_name"], spec["train_origins"], cfg, cache_dir=cache_dir, force_rebuild=args.force_rebuild_cache, progress=(not args.quiet_progress)),
                    "valid_samples": load_task_split_samples(task_package, spec["valid_split_name"], spec["valid_origins"], cfg, cache_dir=cache_dir, force_rebuild=args.force_rebuild_cache, progress=(not args.quiet_progress)),
                })
            task_obj = {
                "task_package": task_package,
                "rolling_fold_samples": rolling_fold_samples,
            }

        task_rows, task_series_rows = run_task_mode(
            task_name=task,
            task_obj=task_obj,
            agg_history_store=hist_store,
            device=device,
            fit_cfg=fit_cfg,
            reg_cfg=reg_cfg,
            quantiles=quantiles,
            mode=args.mode,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_heads=args.num_heads,
            num_sab_layers=args.num_sab_layers,
            num_seeds=args.num_seeds,
            num_inducing_points=args.num_inducing_points,
            recent_window=args.recent_window,
            lag_windows=lag_windows,
            stat_windows=stat_windows,
            monotone_quantiles=monotone_quantiles,
            cnn_num_blocks=args.cnn_num_blocks,
            cnn_kernel_size=args.cnn_kernel_size,
            model_names=model_names,
            wandb_run=wandb_run,
            return_series_rows=True,
        )
        all_rows.append(task_rows)
        if task_series_rows is not None and not task_series_rows.empty:
            all_series_rows.append(task_series_rows)

        task_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        task_rows.to_csv(os.path.join(task_dir, f"proposed_{args.mode}_rows.csv"), index=False)
        if task_series_rows is not None and not task_series_rows.empty:
            task_series_rows.to_csv(os.path.join(task_dir, f"proposed_{args.mode}_series_rows.csv"), index=False)

        del task_obj, task_package, hist_store, agg_daily
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_rows_df = pd.concat(all_rows, axis=0, ignore_index=True)
    all_rows_df.to_csv(os.path.join(args.output_dir, f"proposed_all_{args.mode}_rows.csv"), index=False)
    if all_series_rows:
        all_series_rows_df = pd.concat(all_series_rows, axis=0, ignore_index=True)
        all_series_rows_df.to_csv(os.path.join(args.output_dir, f"proposed_all_{args.mode}_series_rows.csv"), index=False)
    else:
        all_series_rows_df = pd.DataFrame()

    point_summary = summarize_point_results(all_rows_df.to_dict("records"))
    point_summary.to_csv(os.path.join(args.output_dir, f"proposed_point_summary_{args.mode}.csv"), index=False)
    point_table = build_main_point_table(point_summary)
    point_table.to_csv(os.path.join(args.output_dir, f"proposed_Table2_main_point_{args.mode}.csv"))

    print("\nProposed-model point summary:")
    print(point_summary)
    print("\nProposed-model main point table:")
    print(point_table)

    if quantiles is not None:
        quant_summary = summarize_quantile_results(all_rows_df.to_dict("records"))
        quant_summary.to_csv(os.path.join(args.output_dir, f"proposed_quantile_summary_{args.mode}.csv"), index=False)
        quant_table = build_main_quantile_table(quant_summary)
        quant_table.to_csv(os.path.join(args.output_dir, f"proposed_Table3_main_quantile_{args.mode}.csv"))
        print("\nProposed-model quantile summary:")
        print(quant_summary)
        print("\nProposed-model main quantile table:")
        print(quant_table)

    if args.benchmark_csv:
        bench_df = pd.read_csv(args.benchmark_csv)
        combined = pd.concat([bench_df, all_rows_df], axis=0, ignore_index=True)
        combined.to_csv(os.path.join(args.output_dir, f"combined_all_{args.mode}_rows.csv"), index=False)

        point_summary_c = summarize_point_results(combined.to_dict("records"))
        point_summary_c.to_csv(os.path.join(args.output_dir, f"combined_point_summary_{args.mode}.csv"), index=False)
        point_table_c = build_main_point_table(point_summary_c)
        point_table_c.to_csv(os.path.join(args.output_dir, f"combined_Table2_main_point_{args.mode}.csv"))

        if quantiles is not None:
            quant_summary_c = summarize_quantile_results(combined.to_dict("records"))
            quant_summary_c.to_csv(os.path.join(args.output_dir, f"combined_quantile_summary_{args.mode}.csv"), index=False)
            quant_table_c = build_main_quantile_table(quant_summary_c)
            quant_table_c.to_csv(os.path.join(args.output_dir, f"combined_Table3_main_quantile_{args.mode}.csv"))

    if wandb_run.active:
        wandb_run.summary_update({
            "output_dir": args.output_dir,
            "cache_dir": cache_dir,
        })
        wandb_run.log_dataframe("proposed_point_summary", point_summary)
        wandb_run.log_dataframe("proposed_all_rows", all_rows_df)
        if not all_series_rows_df.empty:
            wandb_run.log_dataframe("proposed_all_series_rows", all_series_rows_df)
        if quantiles is not None:
            wandb_run.log_dataframe("proposed_quantile_summary", quant_summary)
        wandb_run.log_artifact_path(args.output_dir, artifact_name=f"m5_proposed_outputs_{args.mode}", artifact_type="results")
        wandb_run.finish()

    print(f"\nDone. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()