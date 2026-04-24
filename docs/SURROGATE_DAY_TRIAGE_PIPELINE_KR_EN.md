# Autoresearch: original decoder vs surrogate-day triage  
# Autoresearch: 기존 디코더 vs surrogate-day 트리아지 비교

This file lives in the **autoresearch-behavior** repo (`https://github.com/limserenahansol/autoresearch-behavior`).  
이 문서는 **autoresearch-behavior** 저장소에 포함되어 있습니다.

---

## English

### What the **original** pipeline does

- **Goal**: Supervised **multi-class** decoding of experimental state from **mouse × day** feature rows (e.g. **Period** 5-class, **Substance** 2-class, **Group** 2-class).
- **Inputs**: `prepare.py` loads `features_day_level.csv` (path in `DATA_PATH`). `run_all_classifiers*.py` builds engineered columns (NaN flags, log transforms, optional `is_active`, pupil peak, etc.).
- **Model**: Best autoresearch config — typically **Stacking** (RF + SVM + logistic meta-learner).
- **Validation**: **Leave-one-mouse-out (LOMO)** — train on 13 mice, test on the held-out mouse; imputation and scaling fit **only on training folds** (no pipeline leakage).
- **Metrics**: `accuracy`, **`per_mouse_acc`**, **`f1_macro`**, confusion matrices, ROCs where applicable.
- **Question answered**: “Given today’s behavior (and pupil/pharma) vector, which **period / drug / group** is this day?”

### What was **added** (surrogate-day triage)

- **New modules**: `surrogate_day_triage_core.py`, `behavior_surrogate_day_triage.py`, `neural_behavior_surrogate_day_triage.py`.
- **Goal**: **Screen** which **columns** (behavior features now, neurons later) help distinguish **observed** mouse–day rows from **surrogate** rows built by swapping in feature vectors from **another day of the same mouse** (a day-resolution control when trial-level time shuffles are not available).
- **Target label**: Binary **real (1) vs surrogate (0)** on **duplicated** rows (balanced); evaluated with **LOMO** on `mouse_key`.
- **Null model**: **Label permutation** shuffle distribution of LOMO accuracy (right-sided *p* = P(null ≥ obs)). **Effect size**: *d*′-style *z* = (obs − mean(null)) / std(null). **Multiple testing**: Benjamini–Hochberg FDR (q = 0.05) and **Holm–Bonferroni** adjusted *p* (family-wise conservative). **Screening gate** (optional, for permissive pre-GLM-style filtering): `d_prime_vs_shuffle ≥ 1` and `p < 0.1` by default (`--dprime-min`, `--screening-alpha`). With tens of thousands of units, Holm can be very strict for gating; the gate emphasizes effect size + shuffle *p* as in common practice.
- **Outputs**: CSVs under `output/surrogate_day_triage_behavior/` and `output/surrogate_day_triage_neural_behavior/` (population summary + per-feature or split per-neuron vs behavior tables). Optional bar chart for top behavior features.
- **Neural path (future)**: `--neural-csv` merged on `mouse_key` + `day_index`; `--demo-synthetic` runs without a neural file.

### How expectations **differ** from the original decoder

| Aspect | Original decoder | Surrogate-day triage |
|--------|------------------|------------------------|
| **Task** | Multiclass **period / substance / group** | Binary **real vs within-mouse random-day surrogate** |
| **Interpretation** | Best global model for a defined label | Which **individual columns** beat shuffle (screening) |
| **Chance baseline** | Depends on class counts (~0.2 for 5-class) | ~**0.5** accuracy if no signal |
| **Data grain** | Same mouse × day table | Same table (+ optional neural columns) |
| **Trial-aligned neural** | Not in this repo | Not required; extension would be a **new** script if you add trial matrices |

**Important**: The triage script does **not** replace `run_all_classifiers.py`. Run it **in addition** when you want a shuffle-controlled screen at **day** resolution.

**Local layout**: If you keep this repo inside `behavior_task/autoresearch/`, that folder is its **own git clone** (nested repository). Clone separately anywhere:  
`git clone https://github.com/limserenahansol/autoresearch-behavior.git`

---

## 한국어 (Korean)

### **기존** 파이프라인이 하는 일

- **목적**: **마우스 × 일(day)** 단위 특징으로 실험 상태를 **다중 클래스**로 예측 (예: **기간**, **약물**, **그룹**).
- **입력**: `prepare.py` → `features_day_level.csv`. `run_all_classifiers*.py`가 가공 특징 생성.
- **모델**: autoresearch 최적 설정 — 보통 **Stacking**.
- **검증**: **LOMO**, 학습 폴드에서만 전처리 적합.
- **질문**: “이 날의 상태는?”

### **추가된** 것 (surrogate-day 트리아지)

- **새 파일**: `surrogate_day_triage_core.py`, `behavior_surrogate_day_triage.py`, `neural_behavior_surrogate_day_triage.py`.
- **목적**: **같은 마우스의 다른 날** 특징으로 만든 **surrogate(대체)** 행과 **실제** 행을 구분하는 데 기여하는 **열(특징/뉴런)** 선별. (trial 단위 시계열이 없을 때의 **일 단위** 통제에 해당.)
- **레이블**: **실제(1) vs surrogate(0)** 이진, **LOMO**.
- **널**: 레이블 **순열**(우측 p), **d_prime_vs_shuffle**(shuffle 평균·표준편차 대비 z), **BH-FDR** + **Holm** 보정 p, **게이트**(기본 d′≥1 & p<0.1, `--dprime-min` 등).
- **결과물**: `output/surrogate_day_triage_behavior/`, `output/surrogate_day_triage_neural_behavior/`.

### 기존과의 **차이** (요약표)

| 항목 | 기존 디코더 | Surrogate-day 트리아지 |
|------|------------|-------------------------|
| **과제** | 기간/약물/그룹 등 다중 클래스 | **실제 vs 같은 마우스 다른 날 surrogate** 이진 |
| **해석** | 전역 분류 모델 | 열별 shuffle 대비 **선별** |

**정리**: 트리아지는 메인 디코더를 **대체하지 않음**. 일 단위 shuffle 통제 **선별**이 필요할 때 **추가** 실행.

**로컬**: `git clone https://github.com/limserenahansol/autoresearch-behavior.git`

---

## Quick commands / 빠른 실행

```bash
python behavior_surrogate_day_triage.py --fast
python neural_behavior_surrogate_day_triage.py --demo-synthetic --fast
```

See root `README.md` for full flags.  
자세한 옵션은 루트 `README.md`를 참고하세요.
