import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_cache_package as acp


def test_cache_plan():
    tmp = Path(tempfile.mkdtemp(prefix="acp_test_"))
    repo = tmp / "repo"
    try:
        for name in ["10pct", "30pct", "50pct", "100pct", "100pct_rolling", "m5_cache"]:
            (repo / "cache" / name).mkdir(parents=True)
            (repo / "cache" / name / "base_panel.pkl").write_bytes(b"\x00")
        (repo / "packaging").mkdir()
        (repo / "packaging" / "package_files").mkdir()
        (repo / "packaging" / "package_files" / "cache_package_README.md").write_text("# cache readme\n")
        cache_root = tmp / "cache_package"
        plan = acp.build_cache_plan(repo, cache_root)
        dsts = {str(item.dst) for item in plan}
        assert str(cache_root / "100pct_rolling") in dsts
        assert str(cache_root / "50pct") in dsts
        assert str(cache_root / "m5_cache") not in dsts
        assert str(cache_root / "README.md") in dsts
        acp.copy_items(plan)
        assert (cache_root / "100pct" / "base_panel.pkl").exists()
        assert (cache_root / "README.md").exists()
        print("test_cache_plan PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_cache_plan()
    print("ALL TESTS PASS")
