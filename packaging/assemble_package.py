"""Assemble the IJF reproducibility package (Package A) from the working tree.

Copies a curated set of files from the code repo and the
M5 data directory into a clean package folder, applies exclusion rules, writes
a manifest and (optionally) zips the result. All stdlib-only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

CODE_INCLUDE = [
    "scripts", "tests", "best_params_all_models.json",
    "appendix_rolling_folds_point.tex", "appendix_rolling_folds_quantile.tex",
    "docs/time_origins.md", "README.md", ".gitignore",
]
DATA_FILES = ["sales_train_evaluation.csv", "sales_train_validation.csv",
              "calendar.csv", "sell_prices.csv"]
WORK_DIRS = ["100pct", "100pct_rolling", "10pct", "10pct_rolling",
             "30pct", "50pct", "figures"]
PACKAGE_FILES = {
    "README.md": "",
    "LICENSE": "",
    "environment.yml": "environment",
    "requirements.txt": "environment",
}


@dataclass(frozen=True)
class CopyItem:
    src: Path
    dst: Path
    kind: str  # "file" | "dir"

    def to_manifest_line(self, package_root: Path) -> str:
        if self.kind == "dir":
            size = sum(f.stat().st_size for f in self.src.rglob("*")
                       if f.is_file() and not prune(f))
        else:
            size = self.src.stat().st_size
        return f"{self.dst.relative_to(package_root).as_posix()}\t{size}\t{self.src}"


def prune(path: Path) -> bool:
    """Return True if the path must be skipped during copy."""
    name = path.name
    if name == "__pycache__" or name.endswith((".pyc", ".pyo")):
        return True
    if name.startswith("optuna_") and name.endswith(".db"):
        return True
    if name in {"m5_experiment_ssh.text", "m5_experiment_ssh.text.pub"}:
        return True
    return False


def build_copy_plan(repo_dir: Path, package_root: Path) -> List[CopyItem]:
    items: List[CopyItem] = []

    data_dir = repo_dir / "data" / "m5"
    for name in DATA_FILES:
        items.append(CopyItem(data_dir / name, package_root / "data" / "m5" / name, "file"))

    for name in CODE_INCLUDE:
        src = repo_dir / name
        items.append(CopyItem(src, package_root / "code" / name,
                              "dir" if src.is_dir() else "file"))

    work = repo_dir / "results" / "work"
    for name in WORK_DIRS:
        items.append(CopyItem(work / name, package_root / "results" / "work" / name, "dir"))

    for name, sub in PACKAGE_FILES.items():
        src = repo_dir / "packaging" / "package_files" / name
        dst_dir = package_root / sub if sub else package_root
        items.append(CopyItem(src, dst_dir / name, "file"))

    return items


def copy_items(items: List[CopyItem]) -> None:
    for item in items:
        item.dst.parent.mkdir(parents=True, exist_ok=True)
        if item.kind == "dir":
            shutil.copytree(item.src, item.dst, dirs_exist_ok=True)
            bad_paths = sorted((p for p in item.dst.rglob("*") if prune(p)),
                               key=lambda p: len(p.parts), reverse=True)
            for bad in bad_paths:
                shutil.rmtree(bad) if bad.is_dir() else bad.unlink()
        else:
            shutil.copy2(item.src, item.dst)


def write_manifest(items: List[CopyItem], package_root: Path, manifest_path: Path) -> None:
    header = "package_path\tsize_bytes\tsource_path\n"
    manifest_path.write_text(
        header + "\n".join(item.to_manifest_line(package_root) for item in items) + "\n",
        encoding="utf-8")


def write_git_sha(sha: str, code_dir: Path) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / ".git_commit_sha.txt").write_text(sha + "\n", encoding="utf-8")


def zip_package(package_root: Path, out_zip: Path) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(package_root.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(package_root))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    root = repo_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=str(repo_dir))
    parser.add_argument("--package-dir", default=str(root / "reproducibility_package"))
    parser.add_argument("--zip", action="store_true", help="Create the zip archive after assembling.")
    parser.add_argument("--clean", action="store_true", help="Delete the package dir first.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying.")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    package_root = Path(args.package_dir)

    plan = build_copy_plan(repo_dir, package_root)
    missing = [i.src for i in plan if not i.src.exists()]
    if missing:
        print("ERROR: missing sources:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    if args.dry_run:
        for item in plan:
            print(item.to_manifest_line(package_root))
        return 0

    if args.clean and package_root.exists():
        shutil.rmtree(package_root)

    copy_items(plan)
    sha = ""
    if shutil.which("git"):
        sha = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    if sha:
        write_git_sha(sha, package_root / "code")
    write_manifest(plan, package_root, package_root / "manifest.tsv")
    print(f"Package assembled at {package_root}: {len(plan)} items (see manifest.tsv)")

    if args.zip:
        out_zip = package_root.parent / f"{package_root.name}_{date.today().isoformat()}.zip"
        zip_package(package_root, out_zip)
        print(f"Zip written: {out_zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
