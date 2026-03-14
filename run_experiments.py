"""
run_experiments.py  --  Experiment Runner & Logger
===================================================
Runs pipeline.py, logs metrics to output/experiment_log.csv,
and saves a snapshot of each pipeline version.

Usage:
    python run_experiments.py                  # run once, auto-increment ID
    python run_experiments.py --id 5           # run with explicit experiment ID
    python run_experiments.py --note "added PCA"  # add a note to the log
"""
import sys
import csv
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
PIPELINE_FILE = ROOT / "pipeline.py"
OUTPUT_DIR = ROOT / "output"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
LOG_FILE = OUTPUT_DIR / "experiment_log.csv"

LOG_HEADER = [
    "exp_id", "timestamp", "per_mouse_acc", "accuracy", "f1_macro",
    "duration_s", "note",
]


def setup_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    SNAPSHOTS_DIR.mkdir(exist_ok=True)


def get_next_id():
    if not LOG_FILE.exists():
        return 0
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        ids = [int(row['exp_id']) for row in reader if row['exp_id'].isdigit()]
    return max(ids) + 1 if ids else 0


def run_pipeline():
    """Import and run pipeline.run(), capturing metrics and timing."""
    sys.path.insert(0, str(ROOT))

    # Force reimport in case pipeline.py was modified between runs
    if 'pipeline' in sys.modules:
        del sys.modules['pipeline']

    start = time.time()
    import pipeline
    result = pipeline.run()
    elapsed = time.time() - start
    return result, elapsed


def log_result(exp_id, result, elapsed, note):
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(LOG_HEADER)
        w.writerow([
            exp_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{result.get('per_mouse_acc', 0):.6f}",
            f"{result.get('accuracy', 0):.6f}",
            f"{result.get('f1_macro', 0):.6f}",
            f"{elapsed:.1f}",
            note,
        ])


def save_snapshot(exp_id):
    dest = SNAPSHOTS_DIR / f"pipeline_exp{exp_id:03d}.py"
    shutil.copy2(PIPELINE_FILE, dest)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, default=None)
    p.add_argument("--note", type=str, default="")
    args = p.parse_args()

    setup_dirs()
    exp_id = args.id if args.id is not None else get_next_id()

    print(f"=== Experiment #{exp_id} ===")
    print(f"Running pipeline.py ...")

    try:
        result, elapsed = run_pipeline()
    except Exception as e:
        print(f"FAILED: {e}")
        log_result(exp_id, {}, 0, f"FAILED: {str(e)[:150]}")
        return

    print(f"Done in {elapsed:.1f}s")
    print(f"  per_mouse_acc = {result['per_mouse_acc']:.6f}")
    print(f"  accuracy      = {result['accuracy']:.6f}")
    print(f"  f1_macro      = {result['f1_macro']:.6f}")

    log_result(exp_id, result, elapsed, args.note)
    snap = save_snapshot(exp_id)
    print(f"  Snapshot saved: {snap.name}")
    print(f"  Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
