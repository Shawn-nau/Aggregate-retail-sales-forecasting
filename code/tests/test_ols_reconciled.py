import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import numpy as np

import m5_benchmarks as mb
from m5_benchmarks import (
    BottomUpGlobalBoostingBenchmark,
    OLSReconciledHistGB,
    build_benchmark_suite,
    clone_benchmark,
)


class FakeRegressor:
    def predict(self, X):
        return np.arange(X.shape[0] * 2, dtype=np.float32).reshape(X.shape[0], 2)


class FakeTop:
    def fit(self, samples):
        return self

    def predict_point(self, samples):
        return np.array([[10.0, 20.0]], dtype=np.float32)


class FakeBottom:
    def fit(self, samples):
        return self

    def predict_child(self, samples):
        return [np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)]


class FakeBottomEmpty:
    def fit(self, samples):
        return self

    def predict_child(self, samples):
        return [np.zeros((0, 2), dtype=np.float32)]


def test_predict_child_returns_blocks_and_point_sums_them():
    mb.build_bottom_level_global_features = lambda samples: (np.zeros((5, 3), dtype=np.float32), [2, 3])
    bench = BottomUpGlobalBoostingBenchmark(random_state=42, clip_predictions=False)
    bench.model_ = FakeRegressor()
    blocks = bench.predict_child([{}, {}])
    assert len(blocks) == 2, blocks
    assert blocks[0].shape == (2, 2) and blocks[1].shape == (3, 2), [b.shape for b in blocks]
    point = bench.predict_point([{}, {}])
    expected = np.vstack([blocks[0].sum(axis=0), blocks[1].sum(axis=0)])
    np.testing.assert_allclose(point, expected)
    print("test_predict_child_returns_blocks_and_point_sums_them PASS")


def test_ols_reconciliation_formula():
    bench = OLSReconciledHistGB(random_state=42)
    bench.top_model = FakeTop()
    bench.bottom_model = FakeBottom()
    preds = bench.predict_point([{}])
    # N=3: (3*[10,20] + [6,6]) / 4 = [9.0, 16.5]
    np.testing.assert_allclose(preds, [[9.0, 16.5]])
    print("test_ols_reconciliation_formula PASS")


def test_ols_reconciliation_zero_children_guard():
    bench = OLSReconciledHistGB(random_state=42)
    bench.top_model = FakeTop()
    bench.bottom_model = FakeBottomEmpty()
    preds = bench.predict_point([{}])
    np.testing.assert_allclose(preds, [[10.0, 20.0]])
    print("test_ols_reconciliation_zero_children_guard PASS")


def test_suite_and_clone():
    suite = build_benchmark_suite(random_state=42)
    assert "Reconcilation" in suite, list(suite)
    cloned = clone_benchmark(suite["Reconcilation"])
    assert isinstance(cloned, OLSReconciledHistGB)
    assert cloned.random_state == 42
    print("test_suite_and_clone PASS")


if __name__ == "__main__":
    test_predict_child_returns_blocks_and_point_sums_them()
    test_ols_reconciliation_formula()
    test_ols_reconciliation_zero_children_guard()
    test_suite_and_clone()
    print("ALL TESTS PASS")
