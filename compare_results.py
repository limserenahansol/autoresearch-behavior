"""
compare_results.py  --  Comparison Table & Visualization
=========================================================
Reads output/experiment_log.csv and generates:
  1. A formatted comparison table (printed to console)
  2. A bar chart saved to output/comparison_chart.png

Usage:
    python compare_results.py
"""
import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
LOG_FILE = ROOT / "output" / "experiment_log.csv"
CHART_FILE = ROOT / "output" / "comparison_chart.png"


def load_log():
    if not LOG_FILE.exists():
        print("No experiment log found. Run some experiments first.")
        return []
    with open(LOG_FILE) as f:
        return list(csv.DictReader(f))


def print_table(rows):
    print()
    print(f"{'Exp':>4}  {'per_mouse_acc':>14}  {'accuracy':>10}  {'f1_macro':>10}  {'time':>6}  {'note'}")
    print("-" * 80)

    best_pma = 0
    for r in rows:
        pma = float(r['per_mouse_acc'])
        marker = " <-- BEST" if pma > best_pma else ""
        if pma > best_pma:
            best_pma = pma
        print(f"  {r['exp_id']:>3}  {pma:>14.6f}  {float(r['accuracy']):>10.6f}  "
              f"{float(r['f1_macro']):>10.6f}  {float(r['duration_s']):>5.1f}s  "
              f"{r.get('note', '')}{marker}")
    print()


def save_chart(rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping chart.")
        return

    ids = [r['exp_id'] for r in rows]
    pma = [float(r['per_mouse_acc']) for r in rows]
    acc = [float(r['accuracy']) for r in rows]
    f1 = [float(r['f1_macro']) for r in rows]
    notes = [r.get('note', '') for r in rows]

    x = np.arange(len(ids))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 1.2), 5))
    ax.bar(x - w, pma, w, label='per_mouse_acc', color='#2196F3')
    ax.bar(x, acc, w, label='accuracy', color='#4CAF50')
    ax.bar(x + w, f1, w, label='f1_macro', color='#FF9800')

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Autoresearch: Experiment Comparison')
    ax.set_xticks(x)
    labels = [f"#{i}\n{n[:20]}" if n else f"#{i}" for i, n in zip(ids, notes)]
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(pma):
        ax.text(i - w, v + 0.01, f"{v:.3f}", ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    print(f"Chart saved: {CHART_FILE}")
    plt.close()


def main():
    rows = load_log()
    if not rows:
        return
    print_table(rows)
    save_chart(rows)


if __name__ == "__main__":
    main()
