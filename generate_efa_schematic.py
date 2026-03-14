"""
generate_efa_schematic.py  --  EFA Calculation Steps (2 figures)
================================================================
Figure A: Steps 1-3 (Data → Correlation → Model)
Figure B: Steps 4-6 (Rotation → Loadings → Scores)

Output:
  output/figures_efa/00a_efa_schematic_part1.png
  output/figures_efa/00b_efa_schematic_part2.png
"""
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, '.')
from prepare_efa import load_mouse_level_data

OUTPUT = Path(__file__).parent / "output" / "figures_efa"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_data():
    features = ['RequirementLast', 'lick_freq_per_min', 'bout_n',
                'rew_n', 'rew_freq_per_min', 'Requirement_speed_per_min']
    delta_pairs = [('Post', 'Pre'), ('Re-exposure', 'Pre'), ('Withdrawal', 'Pre')]
    X_raw, mice, groups, col_names = load_mouse_level_data(
        features=features, delta_pairs=delta_pairs
    )
    nan_frac = np.mean(np.isnan(X_raw), axis=0)
    keep = nan_frac < 0.50
    X_raw = X_raw[:, keep]
    col_names = [c for c, k in zip(col_names, keep) if k]
    X = X_raw.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            med = np.nanmedian(X[:, j])
            X[mask, j] = med if np.isfinite(med) else 0.0
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, mice, groups, col_names


def short_name(c):
    name = c
    for old, new in [('Requirement_speed_per_min', 'Req_speed'),
                     ('RequirementLast', 'ReqLast'),
                     ('lick_freq_per_min', 'lick_freq'),
                     ('rew_freq_per_min', 'rew_freq')]:
        name = name.replace(old, new)
    name = (name.replace('_Re-exposure-Pre', ' (R-Pre)')
                .replace('_Post-Pre', ' (P-Pre)')
                .replace('_Withdrawal-Pre', ' (W-Pre)'))
    return name


def figure_part1(X_scaled, mice, groups, col_names):
    """Steps 1-3: Data → Correlation → Model"""
    R = np.corrcoef(X_scaled.T)
    snames = [short_name(c) for c in col_names]

    fig, axes = plt.subplots(1, 3, figsize=(28, 10),
                              gridspec_kw={'width_ratios': [1.0, 1.2, 0.8]})

    # ── STEP 1: Delta Matrix ────────────────────────────────────────
    ax = axes[0]
    show_n = min(10, len(mice))
    show_f = min(12, len(col_names))
    im = ax.imshow(X_scaled[:show_n, :show_f], aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(show_n))
    mlabels = [f'{mice[i]}  ({"Active" if groups[i]=="Active" else "Passive"})'
               for i in range(show_n)]
    ax.set_yticklabels(mlabels, fontsize=10)
    ax.set_xticks(range(show_f))
    ax.set_xticklabels(snames[:show_f], fontsize=8, rotation=55, ha='right')
    ax.set_title('STEP 1: Scaled Delta Matrix\n(14 mice × 18 delta-features)',
                 fontsize=16, fontweight='bold', color='#1565C0', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.6, label='z-score', pad=0.02)

    ax.text(0.5, -0.28,
            'Each cell = how much this mouse\'s behavior\n'
            'changed from Pre to that phase (z-scored)',
            transform=ax.transAxes, ha='center', fontsize=11,
            fontstyle='italic', color='#555')

    # ── STEP 2: Correlation Matrix ──────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(R[:show_f, :show_f], aspect='equal', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(show_f))
    ax.set_xticklabels(snames[:show_f], fontsize=8, rotation=55, ha='right')
    ax.set_yticks(range(show_f))
    ax.set_yticklabels(snames[:show_f], fontsize=8)

    for i in range(show_f):
        for j in range(show_f):
            val = R[i, j]
            if abs(val) > 0.3:
                color = 'white' if abs(val) > 0.65 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    ax.set_title('STEP 2: Correlation Matrix R (18×18)\n'
                 '"Which variables move together across mice?"',
                 fontsize=16, fontweight='bold', color='#1565C0', pad=15)
    plt.colorbar(im2, ax=ax, shrink=0.6, label='Correlation r', pad=0.02)

    # Draw boxes around correlated blocks
    from matplotlib.patches import Rectangle
    # Post-Pre block (0-5)
    ax.add_patch(Rectangle((-0.5, -0.5), 6, 6, fill=False,
                            edgecolor='#FF9800', linewidth=3, linestyle='--'))
    ax.text(2.5, -1.2, 'Post-Pre\nblock', ha='center', fontsize=9,
            color='#FF9800', fontweight='bold')
    # Reexp-Pre block (6-11)
    if show_f > 6:
        ax.add_patch(Rectangle((5.5, 5.5), min(6, show_f - 6), min(6, show_f - 6),
                                fill=False, edgecolor='#E53935', linewidth=3, linestyle='--'))

    ax.text(0.5, -0.28,
            'Red/blue blocks = groups of correlated variables\n'
            'These blocks become the factors in EFA',
            transform=ax.transAxes, ha='center', fontsize=11,
            fontstyle='italic', color='#555')

    # ── STEP 3: Model Equation ──────────────────────────────────────
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(5, 9.5, 'STEP 3:\nEFA Model', fontsize=18, fontweight='bold',
            color='#1565C0', ha='center', va='top')

    box_text = (
        "The model assumes:\n\n"
        "Each variable = weighted sum\n"
        "of hidden factors + noise\n\n"
        "  Xᵢ = Lᵢ₁×F₁ + Lᵢ₂×F₂ + εᵢ\n\n"
        "In matrix form:\n\n"
        "  R  ≈  L × Lᵀ  +  Ψ\n\n"
        "R = correlation matrix\n"
        "L = loading matrix\n"
        "    (what we solve for!)\n"
        "Ψ = noise (diagonal)\n\n"
        "Algorithm iterates until\n"
        "L×Lᵀ + Ψ  matches  R"
    )
    ax.text(5, 4.5, box_text, fontsize=12, va='center', ha='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#E3F2FD',
                      edgecolor='#1565C0', linewidth=2))

    # ── Arrows between panels ────────────────────────────────────────
    for i in range(2):
        fig.text(0.335 + i * 0.30, 0.55, '→', fontsize=50, fontweight='bold',
                 color='#FF6F00', ha='center', va='center')

    fig.suptitle('How EFA Works — Part 1: Data to Model',
                 fontsize=22, fontweight='bold', color='#0D47A1', y=1.02)

    plt.tight_layout(w_pad=4)
    fig.savefig(OUTPUT / '00a_efa_schematic_part1.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 00a_efa_schematic_part1.png")


def figure_part2(X_scaled, mice, groups, col_names):
    """Steps 4-6: Rotation → Loadings → Scores"""
    snames = [short_name(c) for c in col_names]

    fa_norot = FactorAnalysis(n_components=2, rotation=None, random_state=42)
    fa_norot.fit(X_scaled)
    loadings_raw = fa_norot.components_.T

    fa = FactorAnalysis(n_components=2, rotation='quartimax', random_state=42)
    scores = fa.fit_transform(X_scaled)
    loadings = fa.components_.T

    colors_feat = []
    for c in col_names:
        if 'Withdrawal' in c:
            colors_feat.append('#1565C0')
        elif 'Re-exposure' in c:
            colors_feat.append('#E53935')
        else:
            colors_feat.append('#FF9800')

    fig, axes = plt.subplots(1, 4, figsize=(32, 9),
                              gridspec_kw={'width_ratios': [1, 1, 0.7, 1]})

    # ── STEP 4a: Before Rotation ────────────────────────────────────
    ax = axes[0]
    ax.scatter(loadings_raw[:, 0], loadings_raw[:, 1], c=colors_feat, s=150,
               edgecolors='black', linewidths=1, zorder=3)
    for i, name in enumerate(snames):
        ax.annotate(name, (loadings_raw[i, 0], loadings_raw[i, 1]),
                    fontsize=8, ha='left', xytext=(6, 4), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Factor 1 loading', fontsize=13)
    ax.set_ylabel('Factor 2 loading', fontsize=13)
    ax.set_title('STEP 4a: BEFORE Rotation\n(variables spread across both factors)',
                 fontsize=15, fontweight='bold', color='#1565C0', pad=12)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=11)

    legend_patches = [
        mpatches.Patch(color='#FF9800', label='Post-Pre'),
        mpatches.Patch(color='#E53935', label='Reexp-Pre'),
        mpatches.Patch(color='#1565C0', label='Withdrawal-Pre'),
    ]
    ax.legend(handles=legend_patches, fontsize=11, loc='lower left')

    # ── STEP 4b: After Rotation ─────────────────────────────────────
    ax = axes[1]
    ax.scatter(loadings[:, 0], loadings[:, 1], c=colors_feat, s=150,
               edgecolors='black', linewidths=1, zorder=3)
    for i, name in enumerate(snames):
        ax.annotate(name, (loadings[i, 0], loadings[i, 1]),
                    fontsize=8, ha='left', xytext=(6, 4), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Factor 1 loading (Withdrawal)', fontsize=13)
    ax.set_ylabel('Factor 2 loading (Re-exposure)', fontsize=13)
    ax.set_title('STEP 4b: AFTER Quartimax Rotation\n(each variable → ONE factor only!)',
                 fontsize=15, fontweight='bold', color='#1565C0', pad=12)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=11)

    # Draw circles around clusters
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((-0.7, 0.08), 0.35, 0.15, angle=0,
                          fill=False, edgecolor='#1565C0', linewidth=2.5, linestyle='--'))
    ax.text(-0.7, -0.02, 'Withdrawal\ncluster', ha='center', fontsize=10,
            color='#1565C0', fontweight='bold')

    ax.add_patch(Ellipse((-0.08, 0.62), 0.35, 0.35, angle=0,
                          fill=False, edgecolor='#E53935', linewidth=2.5, linestyle='--'))
    ax.text(0.15, 0.82, 'Re-exposure\n+ Post cluster', ha='center', fontsize=10,
            color='#E53935', fontweight='bold')

    # ── STEP 5: Final Loadings Heatmap ──────────────────────────────
    ax = axes[2]
    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(len(snames)))
    ax.set_yticklabels(snames, fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['F1\n(Withdrawal)', 'F2\n(Re-exposure)'], fontsize=12, fontweight='bold')
    for i in range(len(col_names)):
        for j in range(2):
            val = loadings[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if abs(val) > 0.4 else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                    color=color, fontweight=weight)
    ax.set_title('STEP 5:\nFinal Loadings',
                 fontsize=15, fontweight='bold', color='#1565C0', pad=12)

    # ── STEP 6: Mouse Scores ────────────────────────────────────────
    ax = axes[3]
    active_mask = groups == 'Active'
    ax.scatter(scores[active_mask, 0], scores[active_mask, 1],
               c='#E53935', marker='o', s=180, edgecolors='black',
               linewidths=1, label='Active (n=6)', zorder=3, alpha=0.85)
    ax.scatter(scores[~active_mask, 0], scores[~active_mask, 1],
               c='#1E88E5', marker='s', s=180, edgecolors='black',
               linewidths=1, label='Passive (n=8)', zorder=3, alpha=0.85)
    for i, m in enumerate(mice):
        short_m = m.split('_')[-1] if '_' in m else m[-6:]
        ax.annotate(short_m, (scores[i, 0], scores[i, 1]),
                    fontsize=9, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Factor 1 Score (Withdrawal Response)', fontsize=13)
    ax.set_ylabel('Factor 2 Score (Re-exposure Response)', fontsize=13)
    ax.set_title('STEP 6: Each Mouse Gets a Score\n= Addiction Index in 2D',
                 fontsize=15, fontweight='bold', color='#1565C0', pad=12)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=11)

    # Quadrant labels
    ax.text(0.97, 0.97, 'High withdrawal resilience\n+ High drug-seeking',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            fontstyle='italic', color='#888',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.7))
    ax.text(0.03, 0.03, 'Low withdrawal resilience\n+ Low drug-seeking',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
            fontstyle='italic', color='#888',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.7))

    # ── Arrows between panels ────────────────────────────────────────
    for pos in [0.265, 0.51, 0.72]:
        fig.text(pos, 0.52, '→', fontsize=45, fontweight='bold',
                 color='#FF6F00', ha='center', va='center')

    fig.suptitle('How EFA Works — Part 2: Rotation to Final Addiction Index',
                 fontsize=22, fontweight='bold', color='#0D47A1', y=1.02)

    plt.tight_layout(w_pad=3)
    fig.savefig(OUTPUT / '00b_efa_schematic_part2.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 00b_efa_schematic_part2.png")


def main():
    X_scaled, mice, groups, col_names = load_data()

    print("Generating EFA schematic (2 figures)...")
    print("\n[Part 1] Steps 1-3: Data -> Correlation -> Model")
    figure_part1(X_scaled, mice, groups, col_names)

    print("\n[Part 2] Steps 4-6: Rotation -> Loadings -> Scores")
    figure_part2(X_scaled, mice, groups, col_names)

    print(f"\nDone! Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
