"""
visualize_decoder.py  --  Generate all presentation-ready figures
==================================================================
Reads prediction CSVs from output/ and generates:

  output/figures/
    confusion_5class.png        - 5-class confusion matrix (heatmap)
    confusion_2class.png        - Morphine vs Water confusion matrix
    confusion_group.png         - Active vs Passive confusion matrix
    per_class_accuracy.png      - Per-class accuracy bars for all 3 tasks
    per_mouse_accuracy.png      - Per-mouse accuracy (each mouse as a dot)
    roc_2class.png              - ROC curve for morphine vs water
    roc_group.png               - ROC curve for Active vs Passive
    autoresearch_improvement.png - Improvement trajectory through experiments
    summary_3tasks.png          - Side-by-side comparison of 3 tasks
    feature_importance.png      - Top features used by the best model

Usage:
    python visualize_decoder.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, get_feature_matrix, ALL_CANDIDATE_FEATURES

OUTPUT = Path(__file__).parent / "output"
FIGDIR = OUTPUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'blue': '#2196F3', 'green': '#4CAF50', 'orange': '#FF9800',
    'red': '#F44336', 'purple': '#9C27B0', 'teal': '#009688',
    'active': '#E53935', 'passive': '#1E88E5',
}


def load_predictions(task):
    path = OUTPUT / f"predictions_{task}.csv"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found. Run run_all_classifiers.py first.")
        return None
    return pd.read_csv(path)


def load_proba(task):
    path = OUTPUT / f"proba_{task}.npy"
    if path.exists():
        return np.load(path)
    return None


# ── 1. Confusion Matrices ──────────────────────────────────────────

def plot_confusion(task, class_order, title, filename):
    df = load_predictions(task)
    if df is None:
        return

    cm = confusion_matrix(df['y_true'], df['y_pred'], labels=class_order)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    disp1 = ConfusionMatrixDisplay(cm, display_labels=class_order)
    disp1.plot(ax=axes[0], cmap='Blues', colorbar=False, values_format='d')
    axes[0].set_title(f'{title}\n(counts)', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('Actual', fontsize=10)

    # Percentage
    disp2 = ConfusionMatrixDisplay(cm_pct, display_labels=class_order)
    disp2.plot(ax=axes[1], cmap='Blues', colorbar=False, values_format='.1f')
    axes[1].set_title(f'{title}\n(% per row)', fontsize=12)
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_ylabel('Actual', fontsize=10)

    overall_acc = np.trace(cm) / cm.sum()
    fig.suptitle(f'Overall accuracy: {overall_acc:.1%}', fontsize=11, y=0.02)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGDIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/{filename}")


# ── 2. Per-Class Accuracy ──────────────────────────────────────────

def plot_per_class_accuracy():
    tasks = [
        ('5class', ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure'], 'Period (5-class)'),
        ('2class', ['morphine', 'water'], 'Substance (2-class)'),
        ('group', ['Active', 'Passive'], 'Group (2-class)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bar_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red'], COLORS['purple']]

    for ax, (task, classes, title) in zip(axes, tasks):
        df = load_predictions(task)
        if df is None:
            continue

        accs = []
        for cls in classes:
            mask = df['y_true'] == cls
            if mask.sum() > 0:
                accs.append(df.loc[mask, 'correct'].mean())
            else:
                accs.append(0)

        colors = bar_colors[:len(classes)]
        bars = ax.bar(range(len(classes)), accs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f'{acc:.1%}', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Per-Class Accuracy for Each Classifier Task', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGDIR / 'per_class_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/per_class_accuracy.png")


# ── 3. Per-Mouse Accuracy ──────────────────────────────────────────

def plot_per_mouse_accuracy():
    tasks = [('5class', 'Period (5-class)'), ('2class', 'Substance (2-class)'), ('group', 'Group')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (task, title) in zip(axes, tasks):
        df = load_predictions(task)
        if df is None:
            continue

        mouse_acc = df.groupby(['mouse_key', 'group'])['correct'].mean().reset_index()
        mouse_acc.columns = ['mouse_key', 'group', 'accuracy']
        mouse_acc = mouse_acc.sort_values('accuracy', ascending=False)

        colors = [COLORS['active'] if g == 'Active' else COLORS['passive']
                  for g in mouse_acc['group']]
        ax.barh(range(len(mouse_acc)), mouse_acc['accuracy'], color=colors, edgecolor='white')
        ax.set_yticks(range(len(mouse_acc)))
        ax.set_yticklabels(mouse_acc['mouse_key'], fontsize=7)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel('Accuracy', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(x=mouse_acc['accuracy'].mean(), color='black', linestyle='--', alpha=0.5)
        ax.text(mouse_acc['accuracy'].mean() + 0.02, len(mouse_acc) - 1,
                f'mean={mouse_acc["accuracy"].mean():.3f}', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['active'], label='Active'),
                       Patch(facecolor=COLORS['passive'], label='Passive')]
    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle('Per-Mouse Accuracy (each bar = one mouse)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGDIR / 'per_mouse_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/per_mouse_accuracy.png")


# ── 4. ROC Curves (2-class tasks) ──────────────────────────────────

def plot_roc(task, class_names, title, filename):
    df = load_predictions(task)
    proba = load_proba(task)
    if df is None or proba is None:
        return

    le = LabelEncoder()
    le.fit(class_names)
    y_true = le.transform(df['y_true'])

    if proba.shape[1] < 2:
        print(f"  [SKIP] ROC for {task}: not enough probability columns")
        return

    pos_proba = proba[:, 1]
    ok = np.isfinite(pos_proba)
    if ok.sum() < 10:
        print(f"  [SKIP] ROC for {task}: too few valid probabilities")
        return

    fpr, tpr, _ = roc_curve(y_true[ok], pos_proba[ok])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=COLORS['blue'], lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance (0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['blue'])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{title}\nAUC = {roc_auc:.3f}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    fig.savefig(FIGDIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/{filename}")


# ── 5. Autoresearch Improvement Trajectory ──────────────────────────

def plot_improvement_trajectory():
    log_path = OUTPUT / "experiment_log.csv"
    if not log_path.exists():
        return

    df = pd.read_csv(log_path, encoding='latin-1')

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(df))
    pma = df['per_mouse_acc'].values

    ax.plot(x, pma, 'o-', color=COLORS['blue'], markersize=8, linewidth=2, label='per_mouse_acc')
    ax.fill_between(x, pma, alpha=0.1, color=COLORS['blue'])

    best_so_far = np.maximum.accumulate(pma)
    ax.plot(x, best_so_far, '--', color=COLORS['red'], linewidth=1.5, label='Best so far')

    for i, (val, note) in enumerate(zip(pma, df['note'])):
        short = str(note)[:25] if pd.notna(note) else ''
        offset = 0.01 if val >= best_so_far[i] - 0.001 else -0.025
        ax.annotate(short, (i, val + offset), fontsize=6, ha='center',
                    rotation=30, color='gray')

    ax.set_xlabel('Experiment #', fontsize=11)
    ax.set_ylabel('per_mouse_acc', fontsize=11)
    ax.set_title('Autoresearch: Improvement Trajectory\n(each point = one experiment)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(min(pma) - 0.05, 1.02)
    ax.grid(alpha=0.2)

    start, end = pma[0], best_so_far[-1]
    ax.annotate(f'Start: {start:.3f}', xy=(0, start), fontsize=9, color=COLORS['blue'],
                fontweight='bold')
    ax.annotate(f'Best: {end:.3f} (+{end - start:.3f})', xy=(len(pma) - 1, end),
                fontsize=9, color=COLORS['red'], fontweight='bold',
                ha='right')

    fig.savefig(FIGDIR / 'autoresearch_improvement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/autoresearch_improvement.png")


# ── 6. Summary: 3 Tasks Side-by-Side ───────────────────────────────

def plot_summary_3tasks():
    summary_path = OUTPUT / "summary_all_classifiers.csv"
    if not summary_path.exists():
        return

    df_orig = pd.read_csv(summary_path)

    pupil_path = OUTPUT.parent / "with_pupil" / "summary_all_classifiers.csv"
    has_pupil = pupil_path.exists()
    if has_pupil:
        df_pupil = pd.read_csv(pupil_path)

    metrics = ['per_mouse_acc', 'accuracy', 'f1_macro']
    metric_labels = ['Per-Mouse Acc', 'Accuracy', 'F1 Macro']
    metric_colors_orig = [COLORS['blue'], COLORS['green'], COLORS['orange']]
    metric_colors_pupil = ['#90CAF9', '#A5D6A7', '#FFCC80']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_orig))

    if has_pupil:
        n_metrics = len(metrics)
        total_bars = n_metrics * 2
        bar_w = 0.11
        gap = 0.03

        for mi, (metric, mlabel, col_o, col_p) in enumerate(
                zip(metrics, metric_labels, metric_colors_orig, metric_colors_pupil)):
            offset_orig = (mi * 2 - total_bars / 2 + 0.5) * (bar_w + gap / 2)
            offset_pupil = offset_orig + bar_w + gap / 2

            vals_o = df_orig[metric].values
            vals_p = df_pupil[metric].values

            label_o = f'{mlabel}' if mi == 0 else mlabel
            label_p = f'{mlabel} +pupil' if mi == 0 else f'{mlabel} +pupil'

            bars_o = ax.bar(x + offset_orig, vals_o, bar_w, color=col_o,
                            edgecolor='white', label=label_o)
            bars_p = ax.bar(x + offset_pupil, vals_p, bar_w, color=col_p,
                            edgecolor='white', label=label_p, hatch='///', alpha=0.85)

            for i in range(len(x)):
                ax.text(x[i] + offset_orig, vals_o[i] + 0.008, f'{vals_o[i]:.3f}',
                        ha='center', fontsize=7, fontweight='bold', color=col_o)
                delta = vals_p[i] - vals_o[i]
                sign = '+' if delta >= 0 else ''
                d_color = '#2E7D32' if delta >= 0 else '#C62828'
                ax.text(x[i] + offset_pupil, vals_p[i] + 0.008, f'{vals_p[i]:.3f}',
                        ha='center', fontsize=7, fontweight='bold', color='#555')
                mid_x = (x[i] + offset_orig + x[i] + offset_pupil) / 2
                ax.text(mid_x, max(vals_o[i], vals_p[i]) + 0.03,
                        f'{sign}{delta:.4f}', ha='center', fontsize=7,
                        color=d_color, fontweight='bold')

        ax.set_title('Classifier Comparison: 3 Decoding Tasks\n'
                     'Solid = original  |  Hatched = + pupil_reward_peak',
                     fontsize=13, fontweight='bold')
    else:
        w = 0.22
        ax.bar(x - w, df_orig['per_mouse_acc'], w, label='per_mouse_acc', color=COLORS['blue'])
        ax.bar(x, df_orig['accuracy'], w, label='accuracy', color=COLORS['green'])
        ax.bar(x + w, df_orig['f1_macro'], w, label='f1_macro', color=COLORS['orange'])
        for i, val in enumerate(df_orig['per_mouse_acc']):
            ax.text(i - w, val + 0.01, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title('Classifier Comparison: 3 Decoding Tasks\n'
                     '(using best autoresearch config: Stacking + NaN indicators + log-transform)',
                     fontsize=13, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(df_orig['task'], fontsize=10)
    ax.set_ylim(0.6, 1.10)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.2)

    fig.savefig(FIGDIR / 'summary_3tasks.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/summary_3tasks.png")


# ── 7. Feature Importance ───────────────────────────────────────────

def plot_feature_importance():
    df = load_data()
    X, mouse_keys, groups, periods, days, feat_names = get_feature_matrix(df, ALL_CANDIDATE_FEATURES)

    is_active = (groups == 'Active').astype(float).reshape(-1, 1)
    X = np.column_stack([X, is_active])
    feat_names = list(feat_names) + ['is_active']

    nan_indicators = np.isnan(X).astype(float)
    nan_with_var = np.std(nan_indicators, axis=0) > 0
    if nan_with_var.any():
        nan_names = [f'nan_{feat_names[i]}' for i in range(len(feat_names))
                     if i < len(nan_with_var) and nan_with_var[i]]
        X = np.column_stack([X, nan_indicators[:, nan_with_var]])
        feat_names = feat_names + nan_names

    skewed = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
              'rew_n', 'rew_freq_per_min', 'Requirement_cum']
    for col in skewed:
        if col in feat_names:
            idx = feat_names.index(col)
            X = np.column_stack([X, np.log1p(np.abs(X[:, idx])).reshape(-1, 1)])
            feat_names = feat_names + [f'log_{col}']

    # Impute NaN
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j]) if np.any(~mask) else 0.0

    le = LabelEncoder()
    y = le.fit_transform(periods)

    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_idx)), importances[top_idx][::-1],
            color=COLORS['blue'], edgecolor='white')
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feat_names[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel('Importance (Gini)', fontsize=11)
    ax.set_title('Top 20 Feature Importances\n(Random Forest, 5-class period prediction)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    fig.savefig(FIGDIR / 'feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/feature_importance.png")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING ALL VISUALIZATION FIGURES")
    print("=" * 60)

    print("\n[1] Confusion matrices...")
    plot_confusion('5class',
                   ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure'],
                   'Period Decoder (5-class)', 'confusion_5class.png')
    plot_confusion('2class', ['morphine', 'water'],
                   'Substance Decoder (Morphine vs Water)', 'confusion_2class.png')
    plot_confusion('group', ['Active', 'Passive'],
                   'Group Decoder (Active vs Passive)', 'confusion_group.png')

    print("\n[2] Per-class accuracy...")
    plot_per_class_accuracy()

    print("\n[3] Per-mouse accuracy...")
    plot_per_mouse_accuracy()

    print("\n[4] ROC curves...")
    plot_roc('2class', ['morphine', 'water'],
             'Substance Decoder: Morphine vs Water', 'roc_2class.png')
    plot_roc('group', ['Active', 'Passive'],
             'Group Decoder: Active vs Passive', 'roc_group.png')

    print("\n[5] Autoresearch improvement trajectory...")
    plot_improvement_trajectory()

    print("\n[6] 3-task summary comparison...")
    plot_summary_3tasks()

    print("\n[7] Feature importance...")
    plot_feature_importance()

    print(f"\nAll figures saved to: {FIGDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
