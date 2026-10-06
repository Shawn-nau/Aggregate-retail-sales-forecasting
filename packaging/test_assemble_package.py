import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_package as ap


def _make_fake_tree():
    tmp = Path(tempfile.mkdtemp(prefix="apkg_test_"))
    repo = tmp / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run.py").write_text("x=1\n")
    (repo / "scripts" / "__pycache__").mkdir()
    (repo / "scripts" / "__pycache__" / "run.cpython-39.pyc").write_bytes(b"\x00")
    (repo / "scripts" / "skip.pyo").write_bytes(b"\x00")
    (repo / "m5_experiment_ssh.text").write_text("SECRET")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("pass\n")
    (repo / "best_params_all_models.json").write_text("{}")
    (repo / "README.md").write_text("# readme\n")
    (repo / ".gitignore").write_text("data/\n")
    (repo / "appendix_rolling_folds_point.tex").write_text("ap\n")
    (repo / "appendix_rolling_folds_quantile.tex").write_text("aq\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "time_origins.md").write_text("# origins\n")
    (repo / "data").mkdir()
    (repo / "data" / "m5").mkdir(parents=True)
    (repo / "data" / "m5" / "sales_train_evaluation.csv").write_text("a,b\n")
    (repo / "data" / "m5" / "calendar.csv").write_text("c,d\n")
    (repo / "data" / "m5" / "sell_prices.csv").write_text("e,f\n")
    (repo / "data" / "m5" / "sales_train_validation.csv").write_text("g,h\n")
    (repo / "cache").mkdir()
    (repo / "cache" / "100pct").mkdir()
    (repo / "cache" / "100pct" / "base_panel.pkl").write_bytes(b"\x00")
    (repo / "outputs").mkdir()
    work = repo / "results" / "work"
    (work / "100pct").mkdir(parents=True)
    (work / "100pct" / "optuna_DeepSets.db").write_bytes(b"\x00")
    (work / "100pct" / "06_paper_tables").mkdir()
    (work / "100pct" / "06_paper_tables" / "Table2_main_point.tex").write_text("t\n")
    (work / "100pct_rolling").mkdir()
    (work / "10pct").mkdir()
    (work / "10pct_rolling").mkdir()
    (work / "30pct").mkdir()
    (work / "50pct").mkdir()
    (work / "figures").mkdir()
    (work / "figures" / "f.png").write_bytes(b"\x00")
    (work / "sanity_debug").mkdir()
    (work / "v-3.tex").write_text("old\n")
    (work / "work.zip").write_bytes(b"\x00")
    (work / "references.bib").write_text("@misc{x}\n")
    (repo / "results" / "holdout").mkdir()
    (repo / "results" / "holdout" / "stale.tex").write_text("stale\n")
    (repo / "packaging").mkdir()
    (repo / "packaging" / "package_files").mkdir()
    for name in ["README.md", "LICENSE", "environment.yml", "requirements.txt"]:
        (repo / "packaging" / "package_files" / name).write_text("pkg:" + name + "\n")
    return tmp, repo


def _clean(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def test_plan_and_copy_and_zip():
    tmp, repo = _make_fake_tree()
    pkg = tmp / "reproducibility_package"
    try:
        plan = ap.build_copy_plan(repo, pkg)
        dsts = {item.dst for item in plan}
        assert pkg / "code" / "scripts" in dsts
        assert pkg / "code" / "tests" in dsts
        assert pkg / "data" / "m5" / "sales_train_evaluation.csv" in dsts
        assert pkg / "results" / "work" / "100pct" in dsts
        assert pkg / "results" / "work" / "figures" in dsts
        assert pkg / "results" / "work" / "sanity_debug" not in dsts
        assert pkg / "results" / "work" / "v-3.tex" not in dsts
        assert pkg / "results" / "work" / "work.zip" not in dsts
        assert pkg / "results" / "holdout" not in dsts
        assert pkg / "code" / "cache" not in dsts
        assert pkg / "code" / "m5_experiment_ssh.text" not in dsts

        ap.copy_items(plan)
        ap.write_git_sha("abc123", pkg / "code")
        assert (pkg / "code" / ".git_commit_sha.txt").read_text().strip() == "abc123"
        # pruning inside copied dirs
        assert not (pkg / "code" / "scripts" / "__pycache__").exists()
        assert not (pkg / "code" / "scripts" / "skip.pyo").exists()
        assert not (pkg / "results" / "work" / "100pct" / "optuna_DeepSets.db").exists()
        assert (pkg / "results" / "work" / "100pct" / "06_paper_tables" / "Table2_main_point.tex").exists()

        manifest = pkg / "manifest.tsv"
        ap.write_manifest(plan, pkg, manifest)
        lines = manifest.read_text(encoding="utf-8").strip().splitlines()
        assert any(ln.startswith("results/work/100pct\t") for ln in lines)

        z = tmp / "pkg.zip"
        ap.zip_package(pkg, z)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        assert "manifest.tsv" in names
        assert "code/.git_commit_sha.txt" in names
        print("test_plan_and_copy_and_zip PASS")
    finally:
        _clean(tmp)


if __name__ == "__main__":
    test_plan_and_copy_and_zip()
    print("ALL TESTS PASS")
