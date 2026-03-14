"""
run_cross_generalization.py
============================
Cross-condition generalization analysis:

Part A  --  Cross-GROUP generalization (Period & Substance decoders)
  1. Train on Active only -> test on Passive only
  2. Train on Passive only -> test on Active only
  3. Baseline: within-group LOMO for each group separately

Part B  --  Cross-PERIOD generalization (Group decoder)
  Train on subset of periods, test on held-out periods.
  All pairwise period transfers + leave-one-period-out.

Part C  --  Trajectory similarity analysis
  - Per-period feature centroids for Active vs Passive
  - Cosine similarity of trajectories
  - Magnitude vs direction decomposition
  - Statistical tests for qualitative vs quantitative differences

Saves all results + figures to output/cross_generalization/
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, ALL_CANDIDATE_FEATURES, evaluate_classification

OUT = Path(__file__).parent / "output" / "cross_generalization"
OUT.mkdir(parents=True, exist_ok=True)

COL_A = '#E53935'
COL_P = '#1E88E5'
COL_CROSS = '#FF9800'
COL_GREEN = '#4CAF50'
PERIOD_ORDER = ['Pre', 'During', 'Post', 'Withdrawal', 'Re-exposure']
PERIOD_COLORS = {'Pre': '#2196F3', 'During': '#FF9800', 'Post': '#4CAF50',
                 'Withdrawal': '#9C27B0', 'Re-exposure': '#F44336'}


# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════

def merge_pupil_peak(df):
    pupil_path = Path(__file__).parent / "output" / "pupil_reward_peak.csv"
    if pupil_path.exists():
        pupil = pd.read_csv(pupil_path)
        pupil['mouse_key'] = pupil['mouse_key'].astype(str)
        pupil['day_index'] = pupil['day_index'].astype(int)
        df = df.merge(pupil[['mouse_key', 'day_index', 'pupil_reward_peak']],
                      on=['mouse_key', 'day_index'], how='left')
    return df


def build_X(df):
    feature_cols = [c for c in ALL_CANDIDATE_FEATURES if c in df.columns]
    if 'pupil_reward_peak' in df.columns and 'pupil_reward_peak' not in feature_cols:
        feature_cols.append('pupil_reward_peak')

    X = df[feature_cols].values.astype(float)
    feat_names = list(feature_cols)

    nan_ind = np.isnan(X).astype(float)
    nan_var = np.std(nan_ind, axis=0) > 0
    if nan_var.any():
        X = np.column_stack([X, nan_ind[:, nan_var]])
        feat_names += [f'nan_{feat_names[i]}' for i in range(len(feature_cols)) if i < len(nan_var) and nan_var[i]]

    skewed = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
              'rew_n', 'rew_freq_per_min', 'Requirement_cum']
    for col in skewed:
        if col in feat_names:
            idx = feat_names.index(col)
            X = np.column_stack([X, np.log1p(np.abs(X[:, idx])).reshape(-1, 1)])
            feat_names.append(f'log_{col}')

    return X, feat_names


def impute_and_scale(X_train, X_test):
    col_med = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        med = col_med[j] if np.isfinite(col_med[j]) else 0.0
        X_train[np.isnan(X_train[:, j]), j] = med
        X_test[np.isnan(X_test[:, j]), j] = med
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def make_model():
    return StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000), cv=3)


def lomo_within(X, y, mouse_keys):
    mice = np.unique(mouse_keys)
    y_pred = np.full_like(y, -1)
    for m in mice:
        te = mouse_keys == m
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = impute_and_scale(X[tr].copy(), X[te].copy())
        mdl = make_model()
        mdl.fit(Xtr, y[tr])
        y_pred[te] = mdl.predict(Xte)
    return y_pred


# ═══════════════════════════════════════════════════════════════════
# PART A: Cross-Group Generalization
# ═══════════════════════════════════════════════════════════════════

def part_a(df, X, groups, periods, mouse_keys):
    print("\n" + "=" * 60)
    print("PART A: Cross-Group Generalization (Period & Substance)")
    print("=" * 60)

    results = []

    for target_name, make_y in [
        ('Period', lambda p: p),
        ('Substance', lambda p: np.array(['morphine' if x in ['During', 'Post', 'Re-exposure']
                                           else 'water' for x in p])),
    ]:
        y_labels = make_y(periods)
        le = LabelEncoder()
        le.fit(sorted(set(y_labels)))
        y = le.transform(y_labels)
        classes = list(le.classes_)

        active_mask = groups == 'Active'
        passive_mask = groups == 'Passive'

        # 1. Train Active -> Test Passive
        Xtr, Xte = impute_and_scale(X[active_mask].copy(), X[passive_mask].copy())
        mdl = make_model()
        mdl.fit(Xtr, y[active_mask])
        pred_a2p = mdl.predict(Xte)
        acc_a2p = accuracy_score(y[passive_mask], pred_a2p)

        # 2. Train Passive -> Test Active
        Xtr2, Xte2 = impute_and_scale(X[passive_mask].copy(), X[active_mask].copy())
        mdl2 = make_model()
        mdl2.fit(Xtr2, y[passive_mask])
        pred_p2a = mdl2.predict(Xte2)
        acc_p2a = accuracy_score(y[active_mask], pred_p2a)

        # 3. Within-group LOMO
        pred_within_a = lomo_within(X[active_mask], y[active_mask], mouse_keys[active_mask])
        valid_a = pred_within_a >= 0
        acc_within_a = accuracy_score(y[active_mask][valid_a], pred_within_a[valid_a]) if valid_a.any() else 0

        pred_within_p = lomo_within(X[passive_mask], y[passive_mask], mouse_keys[passive_mask])
        valid_p = pred_within_p >= 0
        acc_within_p = accuracy_score(y[passive_mask][valid_p], pred_within_p[valid_p]) if valid_p.any() else 0

        print(f"\n  {target_name} decoder:")
        print(f"    Within Active (LOMO):       {acc_within_a:.3f}")
        print(f"    Within Passive (LOMO):      {acc_within_p:.3f}")
        print(f"    Active -> Passive (cross):  {acc_a2p:.3f}")
        print(f"    Passive -> Active (cross):  {acc_p2a:.3f}")

        results.append({
            'target': target_name, 'classes': classes, 'le': le,
            'acc_within_a': acc_within_a, 'acc_within_p': acc_within_p,
            'acc_a2p': acc_a2p, 'acc_p2a': acc_p2a,
            'y_passive': y[passive_mask], 'pred_a2p': pred_a2p,
            'y_active': y[active_mask], 'pred_p2a': pred_p2a,
        })

    # ── Figure: Cross-group accuracy comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, r in zip(axes, results):
        labels = ['Within\nActive\n(LOMO)', 'Within\nPassive\n(LOMO)',
                  'Active->\nPassive', 'Passive->\nActive']
        vals = [r['acc_within_a'], r['acc_within_p'], r['acc_a2p'], r['acc_p2a']]
        colors = [COL_A, COL_P, COL_CROSS, COL_CROSS]
        hatches = ['', '', '///', '\\\\\\']

        bars = ax.bar(range(4), vals, color=colors, edgecolor='gray', width=0.6)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

        n_classes = len(r['classes'])
        chance = 1.0 / n_classes
        ax.axhline(y=chance, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(3.4, chance + 0.01, f'chance={chance:.2f}', fontsize=8, color='gray')

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{r["target"]} Decoder', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Cross-Group Generalization:\nDo Active & Passive mice share the same behavioral signatures?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / '01_cross_group_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_cross_group_accuracy.png")

    # ── Confusion matrices for cross-group ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for col, r in enumerate(results):
        for row, (y_true, y_pred, title_sfx) in enumerate([
            (r['y_passive'], r['pred_a2p'], 'Active -> Passive'),
            (r['y_active'], r['pred_p2a'], 'Passive -> Active'),
        ]):
            ax = axes[row, col]
            cm = confusion_matrix(y_true, y_pred, labels=range(len(r['classes'])))
            cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
            disp = ConfusionMatrixDisplay(cm_pct, display_labels=r['classes'])
            disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='.1f')
            acc = accuracy_score(y_true, y_pred)
            ax.set_title(f'{r["target"]}: {title_sfx}\nacc={acc:.1%}',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)

    fig.suptitle('Cross-Group Confusion Matrices (% per row)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / '02_cross_group_confusion.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_cross_group_confusion.png")

    return results


# ═══════════════════════════════════════════════════════════════════
# PART B: Cross-Period Generalization (Group decoder)
# ═══════════════════════════════════════════════════════════════════

def part_b(df, X, groups, periods, mouse_keys):
    print("\n" + "=" * 60)
    print("PART B: Cross-Period Generalization (Group Decoder)")
    print("=" * 60)

    le = LabelEncoder()
    y = le.fit_transform(groups)

    # Remove is_active-like features (would leak group info)
    # Already not in X since build_X doesn't add is_active

    transfer_results = []

    # Pairwise: train on period A, test on period B
    for p_train in PERIOD_ORDER:
        for p_test in PERIOD_ORDER:
            if p_train == p_test:
                continue
            tr_mask = periods == p_train
            te_mask = periods == p_test
            if tr_mask.sum() < 4 or te_mask.sum() < 4:
                continue
            if len(np.unique(y[tr_mask])) < 2:
                continue

            Xtr, Xte = impute_and_scale(X[tr_mask].copy(), X[te_mask].copy())
            mdl = make_model()
            mdl.fit(Xtr, y[tr_mask])
            pred = mdl.predict(Xte)
            acc = accuracy_score(y[te_mask], pred)
            transfer_results.append({
                'train': p_train, 'test': p_test, 'accuracy': acc,
                'n_train': int(tr_mask.sum()), 'n_test': int(te_mask.sum()),
            })
            print(f"  Train={p_train:15s} -> Test={p_test:15s}: acc={acc:.3f}")

    # Within-period LOMO
    within_results = []
    for p in PERIOD_ORDER:
        mask = periods == p
        if mask.sum() < 4 or len(np.unique(y[mask])) < 2:
            within_results.append({'period': p, 'accuracy': np.nan})
            continue
        pred = lomo_within(X[mask], y[mask], mouse_keys[mask])
        valid = pred >= 0
        acc = accuracy_score(y[mask][valid], pred[valid]) if valid.any() else np.nan
        within_results.append({'period': p, 'accuracy': acc})
        print(f"  Within {p:15s} (LOMO): acc={acc:.3f}")

    # ── Figure: Transfer matrix heatmap ──
    n_p = len(PERIOD_ORDER)
    mat = np.full((n_p, n_p), np.nan)
    for r in transfer_results:
        i = PERIOD_ORDER.index(r['train'])
        j = PERIOD_ORDER.index(r['test'])
        mat[i, j] = r['accuracy']
    for r in within_results:
        i = PERIOD_ORDER.index(r['period'])
        mat[i, i] = r['accuracy']

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(n_p))
    ax.set_xticklabels(PERIOD_ORDER, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(PERIOD_ORDER, fontsize=10)
    ax.set_xlabel('Test Period', fontsize=12)
    ax.set_ylabel('Train Period', fontsize=12)

    for i in range(n_p):
        for j in range(n_p):
            val = mat[i, j]
            if np.isfinite(val):
                color = 'white' if val < 0.5 or val > 0.85 else 'black'
                label = f'{val:.2f}' if i != j else f'{val:.2f}\n(LOMO)'
                ax.text(j, i, label, ha='center', va='center', fontsize=9,
                        color=color, fontweight='bold' if i == j else 'normal')

    plt.colorbar(im, ax=ax, label='Group Decoding Accuracy', shrink=0.8)
    ax.set_title('Cross-Period Generalization: Group Decoder (Active vs Passive)\n'
                 'Diagonal = within-period LOMO, off-diagonal = cross-period transfer',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / '03_cross_period_transfer_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_cross_period_transfer_matrix.png")

    # ── Figure: Bar chart of within vs cross-period ──
    fig, ax = plt.subplots(figsize=(12, 5))
    within_acc = [r['accuracy'] for r in within_results]
    cross_mean = []
    for i, p in enumerate(PERIOD_ORDER):
        cross_vals = [mat[j, i] for j in range(n_p) if j != i and np.isfinite(mat[j, i])]
        cross_mean.append(np.mean(cross_vals) if cross_vals else np.nan)

    x = np.arange(n_p)
    w = 0.32
    bars1 = ax.bar(x - w/2, within_acc, w, label='Within-period (LOMO)',
                    color=[PERIOD_COLORS[p] for p in PERIOD_ORDER], edgecolor='white')
    bars2 = ax.bar(x + w/2, cross_mean, w, label='Cross-period transfer (mean)',
                    color=[PERIOD_COLORS[p] for p in PERIOD_ORDER], edgecolor='white',
                    hatch='///', alpha=0.6)

    for i, (v1, v2) in enumerate(zip(within_acc, cross_mean)):
        if np.isfinite(v1):
            ax.text(i - w/2, v1 + 0.02, f'{v1:.2f}', ha='center', fontsize=9, fontweight='bold')
        if np.isfinite(v2):
            ax.text(i + w/2, v2 + 0.02, f'{v2:.2f}', ha='center', fontsize=9, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(n_p - 0.5, 0.52, 'chance=0.50', fontsize=8, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_ORDER, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Group Decoding Accuracy', fontsize=11)
    ax.set_title('Group Decoder: Within-Period vs Cross-Period Transfer',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / '04_cross_period_within_vs_cross.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_cross_period_within_vs_cross.png")

    return transfer_results, within_results


# ═══════════════════════════════════════════════════════════════════
# PART C: Trajectory Similarity (Magnitude vs Direction)
# ═══════════════════════════════════════════════════════════════════

def part_c(df, X, groups, periods, feat_names):
    print("\n" + "=" * 60)
    print("PART C: Trajectory Similarity Analysis")
    print("=" * 60)

    # Use original behavioral + pupil features only (not NaN indicators or log)
    core_feats = [c for c in ALL_CANDIDATE_FEATURES if c in df.columns]
    if 'pupil_reward_peak' in df.columns:
        core_feats.append('pupil_reward_peak')
    X_core = df[core_feats].values.astype(float)

    # Impute with global median
    for j in range(X_core.shape[1]):
        mask = np.isnan(X_core[:, j])
        if mask.any():
            med = np.nanmedian(X_core[:, j])
            X_core[mask, j] = med if np.isfinite(med) else 0.0

    sc = StandardScaler()
    X_z = sc.fit_transform(X_core)

    # Compute per-group per-period centroids
    centroids_a = {}
    centroids_p = {}
    for p in PERIOD_ORDER:
        mask_a = (groups == 'Active') & (periods == p)
        mask_p = (groups == 'Passive') & (periods == p)
        if mask_a.sum() > 0:
            centroids_a[p] = np.mean(X_z[mask_a], axis=0)
        if mask_p.sum() > 0:
            centroids_p[p] = np.mean(X_z[mask_p], axis=0)

    common_periods = sorted(set(centroids_a.keys()) & set(centroids_p.keys()),
                            key=lambda x: PERIOD_ORDER.index(x))

    # Cosine similarity between Active and Passive centroids at each period
    cos_sims = {}
    euclid_dists = {}
    for p in common_periods:
        cos_sims[p] = 1 - cosine(centroids_a[p], centroids_p[p])
        euclid_dists[p] = np.linalg.norm(centroids_a[p] - centroids_p[p])
        print(f"  {p:15s}: cosine_sim={cos_sims[p]:.3f}  euclidean_dist={euclid_dists[p]:.2f}")

    # Trajectory vectors (phase-to-phase changes)
    traj_a = []
    traj_p = []
    traj_labels = []
    for i in range(len(common_periods) - 1):
        p1, p2 = common_periods[i], common_periods[i + 1]
        if p1 in centroids_a and p2 in centroids_a:
            traj_a.append(centroids_a[p2] - centroids_a[p1])
        if p1 in centroids_p and p2 in centroids_p:
            traj_p.append(centroids_p[p2] - centroids_p[p1])
        traj_labels.append(f'{p1[:4]}->{p2[:4]}')

    traj_cos = []
    traj_mag_ratio = []
    for va, vp, lab in zip(traj_a, traj_p, traj_labels):
        cs = 1 - cosine(va, vp) if np.linalg.norm(va) > 0 and np.linalg.norm(vp) > 0 else 0
        mr = np.linalg.norm(va) / max(np.linalg.norm(vp), 1e-10)
        traj_cos.append(cs)
        traj_mag_ratio.append(mr)
        print(f"  Transition {lab}: cos_sim={cs:.3f}  mag_ratio(A/P)={mr:.2f}")

    # ── Figure 5: Centroid similarity across periods ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = range(len(common_periods))
    ax1.bar(x, [cos_sims[p] for p in common_periods],
            color=[PERIOD_COLORS[p] for p in common_periods], edgecolor='white', width=0.6)
    for i, p in enumerate(common_periods):
        ax1.text(i, cos_sims[p] + 0.02, f'{cos_sims[p]:.3f}', ha='center',
                 fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_periods, fontsize=10, rotation=20, ha='right')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Cosine Similarity', fontsize=11)
    ax1.set_title('Active vs Passive: Feature Profile Similarity\n'
                  '(1.0 = identical direction, 0.0 = orthogonal)',
                  fontsize=11, fontweight='bold')
    ax1.axhline(y=0.9, color='green', linestyle=':', alpha=0.5)
    ax1.text(len(common_periods) - 1, 0.92, 'high similarity', fontsize=8, color='green', ha='right')
    ax1.grid(axis='y', alpha=0.2)

    ax2.bar(x, [euclid_dists[p] for p in common_periods],
            color=[PERIOD_COLORS[p] for p in common_periods], edgecolor='white', width=0.6)
    for i, p in enumerate(common_periods):
        ax2.text(i, euclid_dists[p] + 0.1, f'{euclid_dists[p]:.2f}', ha='center',
                 fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_periods, fontsize=10, rotation=20, ha='right')
    ax2.set_ylabel('Euclidean Distance', fontsize=11)
    ax2.set_title('Active vs Passive: Feature Profile Distance\n'
                  '(larger = groups more different)',
                  fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.2)

    fig.suptitle('Are Active & Passive mice in the same "behavioral space" at each period?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / '05_centroid_similarity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_centroid_similarity.png")

    # ── Figure 6: Trajectory direction & magnitude comparison ──
    if len(traj_labels) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        x = range(len(traj_labels))
        colors_t = ['#607D8B'] * len(traj_labels)

        ax1.bar(x, traj_cos, color=colors_t, edgecolor='white', width=0.6)
        for i, v in enumerate(traj_cos):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(traj_labels, fontsize=10)
        ax1.set_ylim(-0.5, 1.15)
        ax1.set_ylabel('Cosine Similarity', fontsize=11)
        ax1.set_title('Trajectory Direction Similarity\n'
                      '(do both groups change in the same "direction"?)',
                      fontsize=11, fontweight='bold')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.axhline(y=0.7, color='green', linestyle=':', alpha=0.5)
        ax1.grid(axis='y', alpha=0.2)

        ax2.bar(x, traj_mag_ratio, color=colors_t, edgecolor='white', width=0.6)
        for i, v in enumerate(traj_mag_ratio):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(traj_labels, fontsize=10)
        ax2.set_ylabel('Magnitude Ratio (Active / Passive)', fontsize=11)
        ax2.set_title('Trajectory Magnitude Ratio\n'
                      '(1.0 = same magnitude, >1 = Active changes more)',
                      fontsize=11, fontweight='bold')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.2)

        fig.suptitle('Same direction, different magnitude? Or qualitatively different trajectories?',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(OUT / '06_trajectory_direction_magnitude.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("  Saved: 06_trajectory_direction_magnitude.png")

    # ── Figure 7: PCA trajectory visualization ──
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_z)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: All data colored by group
    ax = axes[0]
    for grp, col, mk in [('Active', COL_A, 'o'), ('Passive', COL_P, 's')]:
        mask = groups == grp
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=col, marker=mk,
                   s=30, alpha=0.3, label=f'{grp} (all days)')

    # Overlay centroids with arrows
    for grp, centroids_dict, col in [('Active', centroids_a, COL_A), ('Passive', centroids_p, COL_P)]:
        pts = []
        for p in common_periods:
            if p in centroids_dict:
                c_pca = pca.transform(centroids_dict[p].reshape(1, -1))[0]
                pts.append((p, c_pca))
                ax.scatter(c_pca[0], c_pca[1], c=col, s=200, marker='D',
                           edgecolors='black', linewidths=1.5, zorder=5)
                ax.annotate(p[:4], (c_pca[0], c_pca[1]), fontsize=8,
                            fontweight='bold', ha='center', va='bottom',
                            xytext=(0, 8), textcoords='offset points', color=col)

        for i in range(len(pts) - 1):
            ax.annotate('', xy=pts[i+1][1], xytext=pts[i][1],
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.5, alpha=0.7))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_title('Behavioral Trajectory in PCA Space\n(diamonds = period centroids, arrows = trajectory)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.15)

    # Panel 2: Centroid trajectories only (cleaner)
    ax = axes[1]
    for grp, centroids_dict, col, mk in [('Active', centroids_a, COL_A, 'D'),
                                           ('Passive', centroids_p, COL_P, 's')]:
        pts_x, pts_y, labels = [], [], []
        for p in common_periods:
            if p in centroids_dict:
                c_pca = pca.transform(centroids_dict[p].reshape(1, -1))[0]
                pts_x.append(c_pca[0])
                pts_y.append(c_pca[1])
                labels.append(p)

        ax.plot(pts_x, pts_y, '-', color=col, linewidth=2.5, alpha=0.7)
        ax.scatter(pts_x, pts_y, c=col, marker=mk, s=150,
                   edgecolors='black', linewidths=1, zorder=5, label=grp)

        for i, lab in enumerate(labels):
            ax.annotate(lab, (pts_x[i], pts_y[i]), fontsize=9,
                        fontweight='bold', ha='left', va='bottom',
                        xytext=(6, 6), textcoords='offset points', color=col)

        for i in range(len(pts_x) - 1):
            ax.annotate('', xy=(pts_x[i+1], pts_y[i+1]),
                        xytext=(pts_x[i], pts_y[i]),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2, alpha=0.8))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_title('Period Centroid Trajectories\nActive (red) vs Passive (blue)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUT / '07_pca_trajectory.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_pca_trajectory.png")

    # ── Figure 8: Per-feature profile comparison ──
    top_feats_idx = list(range(min(10, len(core_feats))))
    top_feats = [core_feats[i] for i in top_feats_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for pi, p in enumerate(common_periods):
        if pi >= len(axes):
            break
        ax = axes[pi]
        mask_a = (groups == 'Active') & (periods == p)
        mask_p = (groups == 'Passive') & (periods == p)

        vals_a = np.mean(X_z[mask_a][:, top_feats_idx], axis=0) if mask_a.sum() > 0 else np.zeros(len(top_feats))
        vals_p = np.mean(X_z[mask_p][:, top_feats_idx], axis=0) if mask_p.sum() > 0 else np.zeros(len(top_feats))

        x_f = np.arange(len(top_feats))
        ax.barh(x_f - 0.2, vals_a, 0.35, color=COL_A, label='Active', alpha=0.8)
        ax.barh(x_f + 0.2, vals_p, 0.35, color=COL_P, label='Passive', alpha=0.8)
        ax.set_yticks(x_f)
        short_names = [f[:18] for f in top_feats]
        ax.set_yticklabels(short_names, fontsize=7)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_title(f'{p}', fontsize=11, fontweight='bold',
                     color=PERIOD_COLORS.get(p, 'black'))
        if pi == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel('Z-scored mean', fontsize=9)
        ax.grid(axis='x', alpha=0.2)

    for pi in range(len(common_periods), len(axes)):
        axes[pi].set_visible(False)

    fig.suptitle('Feature Profiles: Active vs Passive at Each Period\n'
                 '(same direction = quantitative difference, opposite = qualitative)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / '08_feature_profiles.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 08_feature_profiles.png")

    return cos_sims, traj_cos, traj_mag_ratio


# ═══════════════════════════════════════════════════════════════════
# Figure 9: Grand summary
# ═══════════════════════════════════════════════════════════════════

def plot_grand_summary(cross_group_results, cos_sims, traj_cos, traj_mag_ratio):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    rows = []
    rows.append(['', 'Question', 'Result', 'Interpretation'])
    rows.append(['A1', 'Period decoder: Active->Passive',
                 f"{cross_group_results[0]['acc_a2p']:.3f}",
                 'High' if cross_group_results[0]['acc_a2p'] > 0.7 else 'Low'])
    rows.append(['A2', 'Period decoder: Passive->Active',
                 f"{cross_group_results[0]['acc_p2a']:.3f}",
                 'High' if cross_group_results[0]['acc_p2a'] > 0.7 else 'Low'])
    rows.append(['A3', 'Substance decoder: Active->Passive',
                 f"{cross_group_results[1]['acc_a2p']:.3f}",
                 'High' if cross_group_results[1]['acc_a2p'] > 0.7 else 'Low'])

    mean_cos = np.mean(list(cos_sims.values()))
    rows.append(['C1', 'Mean centroid cosine similarity',
                 f'{mean_cos:.3f}',
                 'Same direction' if mean_cos > 0.7 else 'Different direction'])

    if traj_cos:
        mean_traj = np.mean(traj_cos)
        mean_mag = np.mean(traj_mag_ratio)
        rows.append(['C2', 'Mean trajectory direction similarity',
                     f'{mean_traj:.3f}',
                     'Same trajectory' if mean_traj > 0.5 else 'Different trajectories'])
        rows.append(['C3', 'Mean magnitude ratio (Active/Passive)',
                     f'{mean_mag:.2f}',
                     'Active > Passive' if mean_mag > 1.3 else
                     'Active < Passive' if mean_mag < 0.7 else 'Similar magnitude'])

    conclusion = 'QUANTITATIVE (same pattern, different magnitude)' if mean_cos > 0.7 else 'QUALITATIVE (different patterns)'
    rows.append(['', 'CONCLUSION', conclusion, ''])

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(4):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=10)

    last_row = len(rows) - 1
    for j in range(4):
        table[last_row, j].set_facecolor('#E8F5E9' if 'QUANTITATIVE' in conclusion else '#FFEBEE')
        table[last_row, j].set_text_props(fontweight='bold', fontsize=10)

    ax.set_title('Grand Summary: Cross-Condition Generalization Analysis',
                 fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    fig.savefig(OUT / '09_grand_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 09_grand_summary.png")


# ═══════════════════════════════════════════════════════════════════
# PART D: Cross-Substance Generalization (Period decoder)
# ═══════════════════════════════════════════════════════════════════

def part_d(df, X, groups, periods, mouse_keys):
    print("\n" + "=" * 60)
    print("PART D: Cross-Substance Generalization (Period Decoder)")
    print("=" * 60)

    substance = np.array([
        'morphine' if p in ['During', 'Post', 'Re-exposure'] else 'water'
        for p in periods
    ])

    morph_mask = substance == 'morphine'
    water_mask = substance == 'water'

    morph_periods = set(periods[morph_mask])
    water_periods = set(periods[water_mask])

    results_d = {}

    # ── 1. Train Morphine -> Test Water (Period decoder) ──
    # Morphine has: During, Post, Re-exposure (3 classes)
    # Water has: Pre, Withdrawal (2 classes)
    # These have NO overlapping classes, so a full period decoder can't transfer.
    # Instead: train a GENERAL period decoder on morphine days, see if it can
    # distinguish Pre vs Withdrawal (even though it never saw them).
    # More meaningful: train "all periods" on morphine mice days, test on water days.

    # Better approach: use the GROUP decoder concept --
    # Can morphine behavioral patterns predict water behavioral patterns?
    # i.e., train period classifier on morphine periods, test on water periods
    # with shared label = substance-relative position (early vs late)

    # Most useful: Train period decoder on ALL samples but only from morphine days,
    # test on water days -- and ask: can it distinguish Pre from Withdrawal?
    # Similarly: train on water days, test: can it distinguish morphine periods?

    # Approach A: Train on morphine (3 classes: During/Post/Reexp) -> test on water
    #   -> This doesn't make sense because labels are disjoint.

    # Approach B (meaningful): For the SUBSTANCE decoder (morphine vs water),
    # train on some periods, test on held-out periods.
    # e.g., Train substance decoder on Post+Withdrawal, test on Pre+Re-exposure

    # Approach C (what user asked): Can a model trained on morphine-period data
    # generalize to water-period data? Frame as binary: "morphine experience" patterns.
    # -> Train group decoder on morphine days only, test on water days only.

    # Let's do MULTIPLE meaningful cross-substance analyses:

    # Analysis 1: Group decoder -- train on morphine periods, test on water periods
    print("\n  [D1] Group decoder: Train morphine periods -> Test water periods")
    le_g = LabelEncoder()
    y_g = le_g.fit_transform(groups)

    Xtr, Xte = impute_and_scale(X[morph_mask].copy(), X[water_mask].copy())
    mdl = make_model()
    if len(np.unique(y_g[morph_mask])) >= 2:
        mdl.fit(Xtr, y_g[morph_mask])
        pred_m2w = mdl.predict(Xte)
        acc_m2w = accuracy_score(y_g[water_mask], pred_m2w)
    else:
        acc_m2w = np.nan
    print(f"    Morphine -> Water: acc={acc_m2w:.3f}")
    results_d['group_morph2water'] = acc_m2w

    print("  [D1b] Group decoder: Train water periods -> Test morphine periods")
    Xtr2, Xte2 = impute_and_scale(X[water_mask].copy(), X[morph_mask].copy())
    mdl2 = make_model()
    if len(np.unique(y_g[water_mask])) >= 2:
        mdl2.fit(Xtr2, y_g[water_mask])
        pred_w2m = mdl2.predict(Xte2)
        acc_w2m = accuracy_score(y_g[morph_mask], pred_w2m)
    else:
        acc_w2m = np.nan
    print(f"    Water -> Morphine: acc={acc_w2m:.3f}")
    results_d['group_water2morph'] = acc_w2m

    # Analysis 2: Substance decoder -- train on some periods, test on others
    # (leave-one-period-out for substance classification)
    print("\n  [D2] Substance decoder: Leave-one-period-out")
    y_sub_labels = substance.copy()
    le_s = LabelEncoder()
    y_s = le_s.fit_transform(y_sub_labels)

    lopo_results = []
    for p_test in PERIOD_ORDER:
        te_mask = periods == p_test
        tr_mask = ~te_mask
        if te_mask.sum() < 2 or len(np.unique(y_s[tr_mask])) < 2:
            continue
        Xtr, Xte = impute_and_scale(X[tr_mask].copy(), X[te_mask].copy())
        mdl = make_model()
        mdl.fit(Xtr, y_s[tr_mask])
        pred = mdl.predict(Xte)
        acc = accuracy_score(y_s[te_mask], pred)
        true_label = 'morphine' if p_test in ['During', 'Post', 'Re-exposure'] else 'water'
        lopo_results.append({'held_out': p_test, 'accuracy': acc, 'true_substance': true_label})
        print(f"    Hold out {p_test:15s} (true={true_label:9s}): acc={acc:.3f}")
    results_d['lopo_substance'] = lopo_results

    # Analysis 3: Single-period train for period decoder (what user asked)
    # Train on Pre only -> predict all other periods
    print("\n  [D3] Period decoder: Train single-period, test all others")
    le_p = LabelEncoder()
    le_p.fit(PERIOD_ORDER)
    y_p = le_p.transform(periods)

    single_period_results = []
    for p_train in PERIOD_ORDER:
        tr_mask = periods == p_train
        te_mask = ~tr_mask
        if tr_mask.sum() < 4:
            continue
        # For single-period training, there's only 1 class -> can't train a classifier
        # Instead: train on p_train vs "everything else" in binary,
        # OR train on all periods but only using data from p_train as one class
        # The user wants: "Pre-trained model can predict other periods?"
        # This means: train a multi-class period decoder using ONLY Pre days,
        # which is impossible (1 class).

        # Meaningful alternative: train on period p_train + one other period,
        # and test whether that 2-class boundary generalizes.
        # Better: frame as "same-vs-different" -- train binary (p_train vs rest),
        # test on each held-out period to see if it's classified as "same" or "different"

        # Most useful: for each training period, train BINARY (this period vs not),
        # then test on each individual period -> get accuracy
        y_bin = (periods == p_train).astype(int)
        if len(np.unique(y_bin[tr_mask])) < 1:
            continue

        # Need both classes in training -> use all data but train binary
        Xtr_all, Xte_dummy = impute_and_scale(X.copy(), X[:1].copy())
        mdl = make_model()
        mdl.fit(Xtr_all, y_bin)

        for p_test in PERIOD_ORDER:
            te_p = periods == p_test
            if te_p.sum() == 0:
                continue
            pred = mdl.predict(Xtr_all[te_p])
            # For p_test == p_train, correct = predicted as 1 (positive)
            # For p_test != p_train, correct = predicted as 0 (negative)
            expected = 1 if p_test == p_train else 0
            acc = np.mean(pred == expected)
            single_period_results.append({
                'train_period': p_train, 'test_period': p_test,
                'accuracy': acc, 'same': p_test == p_train
            })

    results_d['single_period'] = single_period_results

    # ── Figure 10: Cross-substance group decoder ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Group decoder morphine <-> water
    ax = axes[0]
    labels = ['Morphine\nperiods\n(LOMO)', 'Water\nperiods\n(LOMO)',
              'Morphine->\nWater', 'Water->\nMorphine']

    # Compute within-substance LOMO group decoder
    acc_within_morph = np.nan
    acc_within_water = np.nan
    if morph_mask.sum() > 4 and len(np.unique(y_g[morph_mask])) >= 2:
        pred_wm = lomo_within(X[morph_mask], y_g[morph_mask], mouse_keys[morph_mask])
        valid = pred_wm >= 0
        acc_within_morph = accuracy_score(y_g[morph_mask][valid], pred_wm[valid]) if valid.any() else np.nan
    if water_mask.sum() > 4 and len(np.unique(y_g[water_mask])) >= 2:
        pred_ww = lomo_within(X[water_mask], y_g[water_mask], mouse_keys[water_mask])
        valid = pred_ww >= 0
        acc_within_water = accuracy_score(y_g[water_mask][valid], pred_ww[valid]) if valid.any() else np.nan

    vals = [acc_within_morph, acc_within_water, acc_m2w, acc_w2m]
    colors = ['#FF9800', '#2196F3', COL_CROSS, COL_CROSS]
    hatches = ['', '', '///', '\\\\\\']
    bars = ax.bar(range(4), vals, color=colors, edgecolor='gray', width=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Group Decoding Accuracy', fontsize=11)
    ax.set_title('Group Decoder: Morphine vs Water Transfer',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)

    # Panel 2: Substance decoder leave-one-period-out
    ax = axes[1]
    if lopo_results:
        held = [r['held_out'] for r in lopo_results]
        accs = [r['accuracy'] for r in lopo_results]
        cols = [PERIOD_COLORS.get(h, 'gray') for h in held]
        bars = ax.bar(range(len(held)), accs, color=cols, edgecolor='white', width=0.6)
        for i, (v, r) in enumerate(zip(accs, lopo_results)):
            ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(held)))
        ax.set_xticklabels(held, fontsize=9, rotation=20, ha='right')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Substance Classification Accuracy', fontsize=11)
        ax.set_title('Substance Decoder: Leave-One-Period-Out\n'
                     '(train on 4 periods, test on held-out)',
                     fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Cross-Substance Generalization', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / '10_cross_substance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 10_cross_substance.png")

    return results_d


# ═══════════════════════════════════════════════════════════════════
# PART E: Single-Period Training -> Test All Others (Period decoder)
# ═══════════════════════════════════════════════════════════════════

def part_e(df, X, groups, periods, mouse_keys):
    print("\n" + "=" * 60)
    print("PART E: Single-Period Train -> Test All Others (Period Decoder)")
    print("=" * 60)
    print("  Training period decoder with LOMO, but only train data from one period")
    print("  + the test mouse's held-out period. Tests: can Pre data predict Post? etc.\n")

    le = LabelEncoder()
    le.fit(PERIOD_ORDER)

    # For each (train_period, test_period) pair, train on all mice's train_period
    # days, test on all mice's test_period days. Multi-class not possible with 1
    # class, so we do pairwise binary: "Is this day from period A or period B?"

    n_p = len(PERIOD_ORDER)
    pair_mat = np.full((n_p, n_p), np.nan)

    pair_results = []
    for i, p_a in enumerate(PERIOD_ORDER):
        for j, p_b in enumerate(PERIOD_ORDER):
            if i == j:
                continue
            mask_a = periods == p_a
            mask_b = periods == p_b
            if mask_a.sum() < 3 or mask_b.sum() < 3:
                continue

            combined_mask = mask_a | mask_b
            X_sub = X[combined_mask].copy()
            y_sub = (periods[combined_mask] == p_b).astype(int)
            mk_sub = mouse_keys[combined_mask]

            # LOMO within this pair
            mice = np.unique(mk_sub)
            y_pred = np.full_like(y_sub, -1)
            for m in mice:
                te = mk_sub == m
                tr = ~te
                if len(np.unique(y_sub[tr])) < 2 or te.sum() == 0:
                    continue
                Xtr, Xte = impute_and_scale(X_sub[tr].copy(), X_sub[te].copy())
                mdl = make_model()
                mdl.fit(Xtr, y_sub[tr])
                y_pred[te] = mdl.predict(Xte)

            valid = y_pred >= 0
            if valid.any():
                acc = accuracy_score(y_sub[valid], y_pred[valid])
            else:
                acc = np.nan
            pair_mat[i, j] = acc
            pair_results.append({'period_A': p_a, 'period_B': p_b, 'accuracy': acc})

    # Also: train on one period's data (as "positive"), test across ALL periods
    # Binary: train "Pre vs all-other-periods-pooled" with LOMO, then report
    # per-period prediction accuracy
    print("  Pairwise period discrimination (LOMO):")
    for r in pair_results:
        print(f"    {r['period_A']:15s} vs {r['period_B']:15s}: acc={r['accuracy']:.3f}")

    # ── Figure 11: Pairwise period discrimination heatmap ──
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(pair_mat, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')

    ax.set_xticks(range(n_p))
    ax.set_xticklabels(PERIOD_ORDER, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(PERIOD_ORDER, fontsize=10)
    ax.set_xlabel('Period B', fontsize=12)
    ax.set_ylabel('Period A', fontsize=12)

    for i in range(n_p):
        for j in range(n_p):
            val = pair_mat[i, j]
            if np.isfinite(val):
                color = 'white' if val > 0.85 or val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')
            elif i == j:
                ax.text(j, i, '--', ha='center', va='center',
                        fontsize=10, color='gray')

    plt.colorbar(im, ax=ax, label='Binary Discrimination Accuracy (LOMO)', shrink=0.8)
    ax.set_title('Pairwise Period Discrimination\n'
                 '"Can I tell Period A from Period B?" (LOMO, binary classifier)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / '11_pairwise_period_discrimination.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 11_pairwise_period_discrimination.png")

    # ── Figure 12: Pre-trained generalization summary ──
    # For user's specific question: "Pre training can predict other periods?"
    # Show Pre vs X accuracy for all X
    fig, ax = plt.subplots(figsize=(10, 5.5))

    pre_idx = PERIOD_ORDER.index('Pre')
    other_periods = [p for p in PERIOD_ORDER if p != 'Pre']
    pre_vs_accs = []
    for p in other_periods:
        j = PERIOD_ORDER.index(p)
        pre_vs_accs.append(pair_mat[pre_idx, j])

    x = range(len(other_periods))
    cols = [PERIOD_COLORS[p] for p in other_periods]
    bars = ax.bar(x, pre_vs_accs, color=cols, edgecolor='white', width=0.6)
    for i, v in enumerate(pre_vs_accs):
        if np.isfinite(v):
            ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(other_periods) - 0.5, 0.52, 'chance', fontsize=9, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Pre vs\n{p}' for p in other_periods], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Binary Discrimination Accuracy', fontsize=11)
    ax.set_title('How distinguishable is Pre (baseline) from each other period?\n'
                 '(LOMO binary classifier)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)

    # Add interpretation annotations
    for i, (p, v) in enumerate(zip(other_periods, pre_vs_accs)):
        if np.isfinite(v):
            if v > 0.85:
                ax.text(i, v - 0.06, 'very\ndifferent', ha='center', fontsize=7,
                        color='white', fontweight='bold')
            elif v > 0.7:
                ax.text(i, v - 0.06, 'different', ha='center', fontsize=7,
                        color='white', fontweight='bold')
            elif v < 0.6:
                ax.text(i, v - 0.06, 'similar', ha='center', fontsize=7,
                        color='white', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT / '12_pre_vs_all_periods.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: 12_pre_vs_all_periods.png")

    # Save results
    pd.DataFrame(pair_results).to_csv(OUT / 'pairwise_period_results.csv', index=False)
    print("  Saved: pairwise_period_results.csv")

    return pair_mat, pair_results


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("CROSS-CONDITION GENERALIZATION ANALYSIS")
    print("=" * 60)

    df = load_data()
    df = merge_pupil_peak(df)
    X, feat_names = build_X(df)

    groups = df['Group'].values.astype(str)
    periods = df['Period'].values.astype(str)
    mouse_keys = df['mouse_key'].values.astype(str)

    cross_group = part_a(df, X, groups, periods, mouse_keys)
    part_b(df, X, groups, periods, mouse_keys)
    cos_sims, traj_cos, traj_mag = part_c(df, X, groups, periods, feat_names)
    part_d(df, X, groups, periods, mouse_keys)
    part_e(df, X, groups, periods, mouse_keys)

    print("\n")
    plot_grand_summary(cross_group, cos_sims, traj_cos, traj_mag)

    # Save numeric results
    pd.DataFrame([
        {'analysis': 'Period: Active->Passive', 'accuracy': cross_group[0]['acc_a2p']},
        {'analysis': 'Period: Passive->Active', 'accuracy': cross_group[0]['acc_p2a']},
        {'analysis': 'Period: Within Active', 'accuracy': cross_group[0]['acc_within_a']},
        {'analysis': 'Period: Within Passive', 'accuracy': cross_group[0]['acc_within_p']},
        {'analysis': 'Substance: Active->Passive', 'accuracy': cross_group[1]['acc_a2p']},
        {'analysis': 'Substance: Passive->Active', 'accuracy': cross_group[1]['acc_p2a']},
        {'analysis': 'Substance: Within Active', 'accuracy': cross_group[1]['acc_within_a']},
        {'analysis': 'Substance: Within Passive', 'accuracy': cross_group[1]['acc_within_p']},
    ]).to_csv(OUT / 'cross_group_results.csv', index=False)

    print(f"\nAll outputs saved to: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
