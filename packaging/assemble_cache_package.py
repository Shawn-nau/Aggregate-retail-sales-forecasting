"""Assemble the supplementary cache package (Package B): plain-folder copy of
the preprocessed sample caches. Copying ~128 GB takes a while — run it in the
background."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import assemble_package as ap

# Re-export the pruned copy implementation so the test can exercise the same
# code path without maintaining a local, prune-skipping duplicate.
copy_items = ap.copy_items

CACHE_DIRS = ["10pct", "30pct", "50pct", "100pct", "100pct_rolling"]


def build_cache_plan(repo_dir: Path, cache_package_root: Path) -> List[ap.CopyItem]:
    items: List[ap.CopyItem] = []
    for name in CACHE_DIRS:
        items.append(ap.CopyItem(repo_dir / "cache" / name, cache_package_root / name, "dir"))
    items.append(ap.CopyItem(
        repo_dir / "packaging" / "package_files" / "cache_package_README.md",
        cache_package_root / "README.md", "file"))
    return items


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    root = repo_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=str(repo_dir))
    parser.add_argument("--cache-package-dir", default=str(root / "cache_package"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    cache_package_root = Path(args.cache_package_dir)

    plan = build_cache_plan(repo_dir, cache_package_root)
    missing = [i.src for i in plan if not i.src.exists()]
    if missing:
        print("ERROR: missing sources:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    if args.dry_run:
        for item in plan:
            print(item.to_manifest_line(cache_package_root))
        return 0

    if args.clean and cache_package_root.exists():
        shutil.rmtree(cache_package_root)

    ap.copy_items(plan)
    ap.write_manifest(plan, cache_package_root, cache_package_root / "manifest.tsv")
    print(f"Cache package assembled at {cache_package_root}: {len(plan)} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
