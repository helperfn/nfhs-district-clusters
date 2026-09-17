"""
Run the whole project end to end, in order.

    python src/run_all.py

Each phase writes its own outputs, so individual scripts can also be re-run on
their own - but the order below is the dependency order and cannot be changed:
baselines needs the k chosen in cluster.py, and evaluate.py needs both.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

PHASES = [
    ("1  download", "download.py"),
    ("2  clean", "clean.py"),
    ("3  eda", "eda.py"),
    ("4  scaling + PCA", "features.py"),
    ("6  clustering", "cluster.py"),        # before baselines: it fixes k
    ("5  baselines", "baselines.py"),
    ("7  evaluation", "evaluate.py"),
    ("7b validation", "validate.py"),
    ("8  cluster profiles", "profile.py"),
    ("9  new-district assignment", "predict.py"),
]


def main() -> None:
    total = time.time()
    for label, script in PHASES:
        print(f"\n{'#' * 78}\n### PHASE {label}  ({script})\n{'#' * 78}")
        start = time.time()
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if result.returncode != 0:
            sys.exit(f"\nPhase {label} failed - stopping.")
        print(f"--- {script} finished in {time.time() - start:.1f}s")
    print(f"\nAll phases complete in {time.time() - total:.1f}s.")
    print("Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
