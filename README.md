# Autoresearch: Mouse Behavior Decoder & Addiction Index

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).  
Adapted for mouse morphine self-administration data: behavioral decoding, addiction index, pupil analysis, and cross-condition generalization.

---

## Overview

This project applies autonomous ML optimization to mouse behavioral data from a morphine self-administration paradigm (14 mice: 6 Active, 8 Passive) across 5 experimental phases.

### Three Analysis Pipelines

| Pipeline | Goal | Key Metric | Best Result |
|----------|------|-----------|-------------|
| **Decoder** | Classify period / substance / group from behavior | per_mouse_acc | 93.3% (period) |
| **Addiction Index (EFA)** | Extract stable latent addiction factors | quality_score | 0.848 |
| **Cross-Generalization** | Test whether behavioral signatures transfer across conditions | accuracy | See results below |

---

## Dataset

| Property | Value |
|----------|-------|
| Mice | 14 (6 Active, 8 Passive) |
| Days | 16 (day 3-18) |
| Rows | 219 (mouse × day) |
| Periods | Pre (3-5), During (6-10), Post (11-13), Withdrawal (14-16), Re-exposure (17-18) |
| Features | 30 behavioral/pupil/pharma + engineered features |
| CV method | Leave-One-Mouse-Out (14 folds) |

**Active mice** = voluntary morphine self-administration via progressive ratio (PR) schedule  
**Passive mice** = yoked morphine delivery (same dose, no behavioral control)

---

## Pipeline 1: Decoder (Classification)

### Results

| Task | Classes | per_mouse_acc | accuracy | f1_macro |
|------|---------|:---:|:---:|:---:|
| **Period** | Pre / During / Post / Withdrawal / Re-exposure | **93.3%** | 93.2% | 92.1% |
| **Substance** | Morphine vs Water | **93.3%** | 93.2% | 92.6% |
| **Group** | Active vs Passive | **81.3%** | 81.3% | 80.9% |

### Best Configuration (found by autoresearch)
- **Model**: Stacking ensemble (Random Forest + SVM + Logistic Regression)
- **Features**: All 30 features + `is_active` indicator + NaN indicators + log-transforms
- **Includes**: `pupil_reward_peak` (peak reward-locked pupil dilation)
- **Improvement**: 58.4% → 93.3% across 9 experiments

### Key Files
```
prepare.py                      # LOCKED: data loader + evaluation functions
pipeline.py                     # Best experiment configuration
run_all_classifiers.py          # Run all 3 tasks (original features)
run_all_classifiers_with_pupil.py  # Run all 3 tasks (+ pupil_reward_peak)
visualize_decoder.py            # Generate presentation figures
compare_results.py              # Experiment comparison
behavior_surrogate_day_triage.py       # Surrogate-day triage (behavior only)
neural_behavior_surrogate_day_triage.py # Same + merged neural CSV
surrogate_day_triage_core.py          # LOMO, shuffle null, Holm, d-prime summaries
surrogate_triage_reporting.py         # Holm on valid per-column p-values
```

### Figures (`output/figures/`)
| File | Description |
|------|-------------|
| `confusion_5class.png` | Period decoder confusion matrix |
| `confusion_2class.png` | Substance decoder confusion matrix |
| `confusion_group.png` | Group decoder confusion matrix |
| `per_class_accuracy.png` | Per-class accuracy for all 3 tasks |
| `per_mouse_accuracy.png` | Per-mouse accuracy colored by group |
| `roc_2class.png` | ROC curve: morphine vs water |
| `roc_group.png` | ROC curve: Active vs Passive |
| `summary_3tasks.png` | 3-task comparison (with/without pupil) |
| `feature_importance.png` | Top 20 features (Gini importance) |
| `autoresearch_improvement.png` | Optimization trajectory |

### Surrogate-day triage (real vs random-within-mouse day)

For **mouse × day** tables (no trial-level timestamps), each **surrogate** row uses the feature vector from **another day of the same mouse** (random-day surrogate), labeled vs the **observed** row. Outputs include shuffle-null mean/std, **d_prime_vs_shuffle**, right-sided permutation *p*, **Holm** and **BH-FDR** passes, and an optional **screening gate** (default *d*′ ≥ 1 and *p* < 0.1). See `surrogate_day_triage_core.py` for details.

| Script | When to use |
|--------|-------------|
| `behavior_surrogate_day_triage.py` | Current pipeline: behavior (+ engineered) features only. |
| `neural_behavior_surrogate_day_triage.py` | After merge: same protocol on **behavior ‖ neural** columns (`--neural-csv` with `mouse_key`, `day_index`, `neuron_*`). |

```bash
python behavior_surrogate_day_triage.py --fast
python behavior_surrogate_day_triage.py --csv path/to/features_day_level.csv --n-shuffles 100
python neural_behavior_surrogate_day_triage.py --neural-csv neural_mouse_day.csv
python neural_behavior_surrogate_day_triage.py --demo-synthetic --fast   # dry run, no neural file
```

Outputs: `output/surrogate_day_triage_behavior/` and `output/surrogate_day_triage_neural_behavior/` (CSVs + optional bar plot).

**CSV columns (triage):** `shuffle_null_mean` / `shuffle_null_std`, **d_prime_vs_shuffle** (effect size vs label-shuffle null), `permutation_p_right_sided`, **Holm**-adjusted *p* + `holm_pass_alpha0.05`, **BH-FDR** `fdr_q0.05_bh`, `exceeds_shuffle_q95`, optional **screening gate** `gate_dprime_and_perm_p` (defaults: `d_prime >= 1` and `p < 0.1`; tune with `--dprime-min`, `--screening-alpha`). Population file includes the same summaries for the pooled model.

```bash
python behavior_surrogate_day_triage.py --fast --dprime-min 1.0 --screening-alpha 0.1
```

**Bilingual note — original decoder vs surrogate-day triage:** [`docs/SURROGATE_DAY_TRIAGE_PIPELINE_KR_EN.md`](docs/SURROGATE_DAY_TRIAGE_PIPELINE_KR_EN.md) (English + 한국어).

---

## Pipeline 2: Addiction Index (EFA)

### Results

| Metric | Value |
|--------|:---:|
| **Quality Score** | **0.848** |
| Stability (split-half) | 0.791 |
| Variance Explained | 84.1% |
| Interpretability | 18/18 features with |loading| > 0.4 |

### Best Configuration (Experiment 10 of 11)
- **Method**: Exploratory Factor Analysis (quartimax rotation)
- **Scaling**: RobustScaler
- **Features**: 6 core behavioral × 3 delta pairs = 18 delta-features
- **Factors**: 2

### Factor Interpretation
| Factor | Driven by | Group p-value | Meaning |
|--------|-----------|:---:|---------|
| Factor 1 | Withdrawal-Pre deltas | p=0.059 | Withdrawal response (craving/dependence) |
| Factor 2 | Re-exposure-Pre deltas | p=0.008** | Re-exposure response (sensitization) |

### Key Files
```
prepare_efa.py                  # LOCKED: data loader + stability evaluation
pipeline_efa.py                 # Best EFA configuration (behavioral only)
pipeline_efa_with_pupil.py      # EFA + pupil_reward_peak (for comparison)
pipeline_efa_pharm.py           # EFA + pharmacological data (for comparison)
visualize_efa.py                # Generate EFA figures
visualize_efa_pharm.py          # EFA + pharma comparison figures
generate_efa_schematic.py       # Step-by-step EFA explanation figure
generate_addiction_score.py     # Composite addiction score per mouse
generate_addiction_trajectory.py # Daily addiction score trajectory
```

### Figures (`output/figures_efa/`)
| File | Description |
|------|-------------|
| `01_factor_loadings.png` | Heatmap of all feature loadings |
| `02_addiction_scores.png` | Per-mouse scores ranked by group |
| `03_factor_scatter.png` | Factor 1 vs Factor 2 (2D space) |
| `04_stability_histogram.png` | Split-half stability distribution |
| `05_improvement_trajectory.png` | Quality score across 11 experiments |
| `06_group_comparison.png` | Box plot: Active vs Passive |
| `07_summary_table.png` | Experiment comparison table |

---

## Pipeline 3: Cross-Condition Generalization

### Key Question
> "Do Active and Passive mice go through the same behavioral trajectory, just at different magnitudes? Or are the patterns qualitatively different?"

### Results Summary

#### Part A: Cross-Group Transfer (Period & Substance Decoders)
| Condition | Period Decoder | Substance Decoder |
|-----------|:---:|:---:|
| Within Active (LOMO) | 88.5% | 95.8% |
| Within Passive (LOMO) | 98.4% | 95.9% |
| **Active → Passive** | **56.1%** | **65.9%** |
| **Passive → Active** | **26.0%** | **36.5%** |

**Conclusion**: Cross-group transfer is poor → groups express phases through **different behavioral patterns**.

#### Part B: Cross-Period Transfer (Group Decoder)
| Period | Within (LOMO) | Cross-period (mean) |
|--------|:---:|:---:|
| Pre | 45.2% (chance) | 54% |
| During | **100%** | 75% |
| Post | **97.6%** | 65% |
| Withdrawal | **100%** | 44% |
| Re-exposure | **100%** | 56% |

**Conclusion**: Groups are indistinguishable at Pre but perfectly separable after morphine. Group-distinguishing features are **period-specific**.

#### Part C: Trajectory Similarity
| Period | Cosine Similarity | Euclidean Distance |
|--------|:---:|:---:|
| Pre | 0.861 (similar) | 1.41 |
| During | 0.144 | 3.01 |
| **Post** | **-0.589 (opposite!)** | 4.93 |
| Withdrawal | 0.084 | 3.57 |
| **Re-exposure** | -0.441 | **7.36 (max)** |

**Conclusion**: **Qualitatively different trajectories.** Groups start similar (Pre: cosine=0.86) but develop opposite behavioral profiles after morphine (Post: cosine=-0.59). Maximum divergence at Re-exposure (distance=7.36).

#### Part D: Cross-Substance Transfer
- Group decoder trained on morphine periods → tested on water periods: **40.7%** (fails)
- Substance decoder leave-one-period-out: Pre and Withdrawal (both water) are **not interchangeable** despite same substance

#### Part E: Pairwise Period Discrimination
- Every period pair is distinguishable with >86% accuracy
- Hardest pair: Withdrawal vs Re-exposure (86.2%)
- Pre vs any period: >99% accuracy → baseline behavior is unique

### Key Files
```
run_cross_generalization.py     # All 5 parts (A-E) + all figures
```

### Figures (`output/cross_generalization/`)
| File | Description |
|------|-------------|
| `01_cross_group_accuracy.png` | Within vs cross-group decoder accuracy |
| `02_cross_group_confusion.png` | Confusion matrices for cross-group transfer |
| `03_cross_period_transfer_matrix.png` | 5×5 heatmap: group decoder period transfer |
| `04_cross_period_within_vs_cross.png` | Within vs cross-period group decoding |
| `05_centroid_similarity.png` | Cosine similarity & distance per period |
| `06_trajectory_direction_magnitude.png` | Direction similarity & magnitude ratio |
| `07_pca_trajectory.png` | PCA visualization of group trajectories |
| `08_feature_profiles.png` | Per-feature Active vs Passive at each period |
| `09_grand_summary.png` | Summary table with conclusion |
| `10_cross_substance.png` | Morphine/water cross-substance transfer |
| `11_pairwise_period_discrimination.png` | 5×5 pairwise period discrimination |
| `12_pre_vs_all_periods.png` | Pre vs each other period |

---

## Pupil Analysis

### Key Files
```
extract_pupil_feature.py        # Extract peak reward-locked pupil dilation
generate_pupil_trajectory.py    # Daily pupil trajectory (% change from Pre)
generate_pupil_timecourse.py    # Within-session pupil timecourse (10s bins)
generate_pupil_event_locked.py  # Reward-locked & lick-locked pupil traces
compare_with_pupil.py           # Decoder & EFA comparison: with vs without pupil
```

### Figures (`output/with_pupil/figures/`)
| File | Description |
|------|-------------|
| `01_decoder_comparison.png` | Decoder: with vs without pupil_reward_peak |
| `02_efa_comparison.png` | EFA: with vs without pupil_reward_peak |
| `03_efa_loadings_with_pupil.png` | EFA loadings heatmap (pupil highlighted) |
| `07_summary_table.png` | Combined comparison table |

### Impact of Adding Pupil
| Pipeline | Without Pupil | + Pupil Peak | Change |
|----------|:---:|:---:|:---:|
| Decoder: Group | 78.9% | **81.3%** | **+2.4%** |
| Decoder: Period | **93.8%** | 93.3% | -0.5% |
| EFA: Quality | **0.848** | 0.766 | -0.082 |

**Conclusion**: Pupil helps group decoder (+2.4%) but hurts EFA stability (too few mice for extra features).

---

## Quick Start

```bash
# 1. Run the best decoder on all 3 tasks
python run_all_classifiers_with_pupil.py

# 2. Generate decoder figures
python visualize_decoder.py

# 3. Run EFA addiction index
python pipeline_efa.py

# 4. Generate EFA figures
python visualize_efa.py

# 5. Run cross-generalization analysis
python run_cross_generalization.py

# 6. Compare with/without pupil
python compare_with_pupil.py
```

### Requirements
```
numpy
pandas
scipy
scikit-learn
matplotlib
```

---

## Complete File Structure

```
autoresearch/
│
├── prepare.py                        # LOCKED: data loader + evaluation
├── prepare_efa.py                    # LOCKED: EFA data loader + stability
├── pipeline.py                       # Best decoder config
├── pipeline_efa.py                   # Best EFA config (behavioral)
├── pipeline_efa_with_pupil.py        # EFA + pupil (comparison)
├── pipeline_efa_pharm.py             # EFA + pharma (comparison)
│
├── run_experiments.py                # Single experiment runner
├── run_all_classifiers.py            # All 3 decoder tasks (original)
├── run_all_classifiers_with_pupil.py # All 3 decoder tasks (+ pupil)
├── run_cross_generalization.py       # Cross-condition analysis (Parts A-E)
├── surrogate_day_triage_core.py      # LOMO + shuffle-null + Holm / d-prime helpers
├── surrogate_triage_reporting.py     # Holm mapping onto valid per-column p-values
├── behavior_surrogate_day_triage.py # Behavior-only surrogate-day triage
├── neural_behavior_surrogate_day_triage.py # Behavior + neural merge, same triage
│
├── extract_pupil_feature.py          # Extract pupil_reward_peak from raw CSV
├── generate_pupil_trajectory.py      # Daily pupil trajectory
├── generate_pupil_timecourse.py      # Within-session pupil timecourse
├── generate_pupil_event_locked.py    # Event-locked pupil traces
├── generate_addiction_score.py       # Composite addiction score
├── generate_addiction_trajectory.py  # Daily addiction score trajectory
├── generate_efa_schematic.py         # EFA explanation figure
│
├── visualize_decoder.py              # Decoder presentation figures
├── visualize_efa.py                  # EFA presentation figures
├── visualize_efa_pharm.py            # EFA + pharma figures
├── compare_results.py                # Experiment comparison
├── compare_with_pupil.py             # With/without pupil comparison
│
├── agent.py                          # Autonomous agent loop
├── program.md                        # Decoder agent instructions
├── program_efa.md                    # EFA agent instructions
├── README.md                         # This file
│
└── output/
    ├── figures/                       # Decoder figures
    ├── figures_efa/                   # EFA figures
    ├── figures_efa_pharm/             # EFA + pharma figures
    ├── figures_pupil/                 # Pupil analysis figures
    ├── cross_generalization/          # Cross-condition figures
    ├── with_pupil/                    # With-pupil comparison
    │   └── figures/
    ├── snapshots/                     # Experiment code snapshots
    ├── experiment_log.csv             # Decoder experiment log
    ├── efa_experiment_log.csv         # EFA experiment log
    ├── summary_all_classifiers.csv    # 3-task summary
    ├── predictions_*.csv              # Per-row predictions
    ├── metrics_*.csv                  # Per-class metrics
    ├── pupil_reward_peak.csv          # Extracted pupil feature
    ├── addiction_scores.csv           # Per-mouse addiction scores
    └── addiction_scores_daily.csv     # Daily addiction scores
```

---

## 한국어 요약

### 이 프로젝트가 하는 일

마우스 모르핀 자가투여 데이터를 사용해 3가지 분석을 수행합니다:

1. **디코더 (분류기)**: 행동 데이터로 실험 기간/약물/그룹을 예측 (93.3% 정확도)
2. **중독 지수 (EFA)**: 안정적인 잠재 중독 요인 추출 (quality=0.848)
3. **교차 일반화**: Active vs Passive 마우스의 행동 궤적이 질적으로 다른지 검증

### 핵심 발견

- **기간 디코더 93.3%**: 행동만으로 어떤 실험 기간인지 거의 완벽하게 예측 가능
- **그룹 디코더 81.3%**: 동공 반응 추가 시 Active vs Passive 구분 향상
- **교차 일반화 실패**: Active 마우스로 훈련한 모델이 Passive에서 작동하지 않음 (56%) → 두 그룹은 **질적으로 다른 행동 패턴**을 보임
- **Pre에서 시작 동일, 이후 영구 분리**: 기저선에서는 그룹 구분 불가 (45%), 모르핀 후 100% 구분 가능. 금단기에도 재수렴하지 않음
- **재노출 시 최대 분리**: Active와 Passive의 행동 프로파일 거리가 재노출 시 최대 (7.36, 기저선의 5.2배)
