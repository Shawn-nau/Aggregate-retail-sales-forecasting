from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


class OptionalDependencyError(RuntimeError):
    pass


def maybe_import_wandb():
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise OptionalDependencyError(
            "wandb is not installed. Install it with: pip install wandb"
        ) from exc
    return wandb


def maybe_import_optuna():
    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise OptionalDependencyError(
            "optuna is not installed. Install it with: pip install optuna"
        ) from exc
    return optuna


def parse_tags(tag_string: str) -> Sequence[str]:
    return tuple(tag.strip() for tag in tag_string.split(',') if tag.strip()) if tag_string else tuple()


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _to_python_scalar(value)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = _to_python_scalar(v)
    return items


class WandbRunWrapper:
    def __init__(
        self,
        enabled: bool,
        project: str = '',
        entity: str = '',
        run_name: str = '',
        tags: Sequence[str] = (),
        config: Optional[Dict[str, Any]] = None,
        mode: str = 'online',
        group: str = '',
        job_type: str = '',
        notes: str = '',
        reinit: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._wandb = None
        self.run = None
        if not self.enabled:
            return
        wandb = maybe_import_wandb()
        kwargs: Dict[str, Any] = {
            'project': project or None,
            'entity': entity or None,
            'name': run_name or None,
            'tags': list(tags) if tags else None,
            'config': make_json_safe(config or {}),
            'mode': mode,
            'group': group or None,
            'job_type': job_type or None,
            'notes': notes or None,
            'reinit': reinit,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._wandb = wandb
        self.run = wandb.init(**kwargs)

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None) -> None:
        if not self.active:
            return
        payload = make_json_safe(metrics)
        if step is not None:
            payload['_step'] = int(step)
        self._wandb.log(payload, commit=commit)

    def define_metric(self, *args, **kwargs) -> None:
        if not self.active:
            return
        self._wandb.define_metric(*args, **kwargs)

    def summary_update(self, values: Dict[str, Any]) -> None:
        if not self.active:
            return
        for k, v in flatten_dict(make_json_safe(values)).items():
            self.run.summary[k] = v

    def log_dataframe(self, name: str, df: pd.DataFrame) -> None:
        if not self.active:
            return
        self.run.log({name: self._wandb.Table(dataframe=df)})

    def log_artifact_path(self, path: str, artifact_name: str, artifact_type: str = 'dataset') -> None:
        if not self.active or not os.path.exists(path):
            return
        artifact = self._wandb.Artifact(name=artifact_name, type=artifact_type)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.active:
            self.run.finish()


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(obj), f, indent=2, sort_keys=True)
