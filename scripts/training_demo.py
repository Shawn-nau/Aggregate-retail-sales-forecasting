from __future__ import annotations

import torch

from run_m5_proposed_models import (
    FitConfig,
    TrainRegularizationConfig,
    RetailAggregateDataset,
    build_model,
    fit_model,
    fit_sample_scalers,
    predict_dataset,
    configure_torch_runtime,
)


def make_synthetic_samples(n_samples: int, t_hist: int, horizon: int, f_hist: int, f_future: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(n_samples):
        n_skus = int(torch.randint(12, 28, (1,), generator=g).item())
        y_hist = torch.poisson(torch.full((n_skus, t_hist, 1), 2.0, dtype=torch.float32, device='cpu')).numpy()
        x_hist = torch.randn(n_skus, t_hist, f_hist, generator=g).numpy().astype('float32')
        x_future = torch.randn(n_skus, horizon, f_future, generator=g).numpy().astype('float32')
        y_target = (torch.tensor(y_hist[:, -horizon:, 0]).sum(dim=0) / max(n_skus, 1) + torch.randn(horizon, generator=g)).numpy().astype('float32')
        samples.append({
            'y_hist': y_hist.astype('float32'),
            'x_hist': x_hist,
            'x_future': x_future,
            'y_target': y_target,
            'meta': {'agg_id': f'synth_{i}', 'origin_time': i},
        })
    return samples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configure_torch_runtime(device, cpu_threads=1)

    t_hist = 56
    horizon = 14
    f_hist = 6
    f_future = 6

    train_samples_raw = make_synthetic_samples(64, t_hist, horizon, f_hist, f_future, seed=1)
    valid_samples_raw = make_synthetic_samples(16, t_hist, horizon, f_hist, f_future, seed=2)
    test_samples_raw = make_synthetic_samples(16, t_hist, horizon, f_hist, f_future, seed=3)

    scalers = fit_sample_scalers(train_samples_raw, use_static=False)
    train_dataset = RetailAggregateDataset(train_samples_raw, scalers=scalers, use_static=False, materialize=True)
    valid_dataset = RetailAggregateDataset(valid_samples_raw, scalers=scalers, use_static=False, materialize=True)
    test_dataset = RetailAggregateDataset(test_samples_raw, scalers=scalers, use_static=False, materialize=True)

    model = build_model(
        model_name='SetTransformer',
        hist_feat_dim=f_hist,
        future_feat_dim=f_future,
        horizon=horizon,
        quantiles=None,
        hidden_dim=32,
        dropout=0.1,
        num_heads=4,
        num_sab_layers=1,
        num_seeds=1,
        num_inducing_points=8,
        recent_window=14,
        lag_windows=(7, 14, 28),
        stat_windows=(7, 28),
        monotone_quantiles=True,
        cnn_num_blocks=1,
        cnn_kernel_size=3,
    )

    fit_cfg = FitConfig(epochs=3, batch_size=8, patience=2, batch_log_interval=2, materialize_datasets=True)
    reg_cfg = TrainRegularizationConfig(use_permutation_consistency=False)

    model, history = fit_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        device=device,
        fit_cfg=fit_cfg,
        quantiles=None,
        reg_cfg=reg_cfg,
    )

    pred_std = predict_dataset(model, test_dataset, device=device, batch_size=8)
    pred_raw = scalers['y_target'].inverse_transform(pred_std)
    print('History keys:', history.keys())
    print('Prediction shape:', pred_raw.shape)


if __name__ == '__main__':
    main()
