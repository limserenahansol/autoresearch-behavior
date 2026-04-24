# Autoresearch: original decoder vs Michelle-style triage  
# Autoresearch: 기존 디코더 vs Michelle-style 트리아지 비교

This file lives in the **autoresearch-behavior** repo (`https://github.com/limserenahansol/autoresearch-behavior`).  
이 문서는 **autoresearch-behavior** 저장소에 포함되어 있습니다.

---

## English

### What the **original** pipeline does

- **Goal**: Supervised **multi-class** decoding of experimental state from **mouse × day** feature rows (e.g. **Period** 5-class, **Substance** 2-class, **Group** 2-class).
- **Inputs**: `prepare.py` loads `features_day_level.csv` (path in `DATA_PATH`). `run_all_classifiers*.py` builds engineered columns (NaN flags, log transforms, optional `is_active`, pupil peak, etc.).
- **Model**: Best autoresearch config — typically **Stacking** (RF + SVM + logistic meta-learner).
- **Validation**: **Leave-one-mouse-out (LOMO)** — train on 13 mice, test on the held-out mouse; imputation and scaling fit **only on training folds** (no pipeline leakage).
- **Metrics**: `accuracy`, **`per_mouse_acc`** (mean of per-mouse hit rates), **`f1_macro`**, confusion matrices, ROCs where applicable.
- **Question answered**: “Given today’s behavior (and pupil/pharma) vector, which **period / drug / group** is this day?”

### What was **added** (Michelle-style adaptation)

- **New modules**: `michelle_style_core.py`, `michelle_style_behavior_triage.py`, `michelle_style_neural_behavior_triage.py`.
- **Goal**: **Triage / screening** — which **columns** (behavior features now, neurons later) help distinguish **“real” mouse–day rows** from **“fake”** rows built by swapping in feature vectors from **another day of the same mouse** (a day-resolution analogue of “real alignment vs random-time control” when you do not have trial-level traces).
- **Target label**: Binary **real (1) vs fake (0)** on **duplicated** rows (balanced); still evaluated with **LOMO** on `mouse_key` so held-out mice are never used for training that mouse’s predictions.
- **Null model**: **Label permutation** shuffle distribution of LOMO accuracy (Michelle-style control). **Benjamini–Hochberg FDR** (q = 0.05) on permutation *p*-values across columns.
- **Outputs**: CSVs under `output/michelle_style_behavior/` and `output/michelle_style_neural_behavior/` (population summary + per-feature or split per-neuron vs behavior tables). Optional bar chart for top behavior features.
- **Neural path (future)**: `--neural-csv` merged on `mouse_key` + `day_index`; `--demo-synthetic` runs without a neural file.

### How expectations **differ** from the original decoder

| Aspect | Original decoder | Michelle-style triage |
|--------|------------------|------------------------|
| **Task** | Multiclass **period / substance / group** | Binary **real vs within-mouse random-day surrogate** |
| **Interpretation** | Best global model for a defined label | Which **individual columns** beat shuffle (screening) |
| **Chance baseline** | Depends on class counts (~0.2 for 5-class) | ~**0.5** accuracy if no signal |
| **Data grain** | Same mouse × day table | Same table (+ optional neural columns) |
| **Trial-aligned neural** | Not in this repo | Not required; extension would be a **new** script if you add trial matrices |

**Important**: The triage script does **not** replace `run_all_classifiers.py`. Run it **in addition** when you want a shuffle-controlled screen analogous to Michelle’s workflow at **day** resolution.

**Local layout**: If you keep this repo inside `behavior_task/autoresearch/`, that folder is its **own git clone** (nested repository). Clone separately anywhere:  
`git clone https://github.com/limserenahansol/autoresearch-behavior.git`

---

## 한국어 (Korean)

### **기존** 파이프라인이 하는 일

- **목적**: **마우스 × 일(day)** 단위 행동(및 동공·약리 등) 특징으로, 실험 상태를 **다중 클래스**로 예측합니다. (예: **기간(Period)** 5-class, **약물(Substance)** 2-class, **그룹(Group)** 2-class.)
- **입력**: `prepare.py`가 `features_day_level.csv`를 읽습니다(`DATA_PATH`). `run_all_classifiers*.py`가 NaN 표시, log 변환, `is_active`, 동공 peak 등 **가공 특징**을 붙입니다.
- **모델**: autoresearch로 고른 최적 설정 — 보통 **Stacking**(RF + SVM + 메타 로지스틱).
- **검증**: **LOMO(마우스 한 마리씩 제외)** — 13마리로 학습, 빠진 1마리로 평가. 결측 대체·스케일링은 **학습 폴드에서만** 적합합니다(파이프라인 누수 없음).
- **지표**: `accuracy`, **`per_mouse_acc`**, **`f1_macro`**, 혼동행렬, ROC 등.
- **답하는 질문**: “이 날의 행동 벡터가 주어졌을 때, **어느 기간/약물/그룹**인가?”

### **추가된** 것 (Michelle 방식을 행동 데이터에 맞게 적용)

- **새 파일**: `michelle_style_core.py`, `michelle_style_behavior_triage.py`, `michelle_style_neural_behavior_triage.py`.
- **목적**: **트리아지(선별)** — 어떤 **열(특징 또는 이후 뉴런)**이 **“실제” 마우스–일 행**과, **같은 마우스의 다른 날** 특징 벡터로 만든 **“가짜(fake)”** 행을 구분하는 데 도움이 되는지 봅니다. (trial 단위 시계열이 없을 때, “실제 정렬 vs 임의 시점”에 대응하는 **일 단위** 대안입니다.)
- **레이블**: **실제(1) vs 가짜(0)** 이진 분류. 행을 두 배로 쌓아 균형을 맞추고, **`mouse_key` 기준 LOMO**로 여전히 **마우스 밖 일반화**를 평가합니다.
- **널 모델**: 레이블 **순열(permutation)** 으로 만든 LOMO 정확도 분포(Michelle의 shuffle 통제와 같은 취지). 열(특징/뉴런)마다 **순열 p값**에 **Benjamini–Hochberg FDR**(q=0.05) 적용.
- **결과물**: `output/michelle_style_behavior/`, `output/michelle_style_neural_behavior/` 아래 CSV(집단 요약 + 특징별 또는 뉴런/행동 분리 표). 행동 특징 상위 막대 그림은 선택.
- **미래 뉴런**: `--neural-csv`로 `mouse_key`, `day_index` 기준 병합; `--demo-synthetic`은 뉴런 파일 없이 파이프라인만 테스트합니다.

### 기존 파이프라인과 **기대되는 차이**

| 항목 | 기존 디코더 | Michelle-style 트리아지 |
|------|------------|-------------------------|
| **과제** | 기간/약물/그룹 등 **다중 클래스** | **실제 vs 같은 마우스 다른 날 surrogate** 이진 |
| **해석** | 정해진 라벨에 대한 **전역 최적 모델** | 열 하나하나가 shuffle보다 나은지 **선별** |
| **우연 수준** | 클래스 수에 따름(5-class면 ~0.2) | 신호 없으면 정확도 ~**0.5** 부근 |
| **데이터 단위** | 동일한 마우스×일 표 | 동일 (+ 선택적 뉴런 열) |
| **trial 정렬 뉴런** | 이 저장소 범위 밖 | 필요 시 **별도 스크립트**로 확장 |

**정리**: 트리아지 스크립트는 `run_all_classifiers.py`를 **대체하지 않습니다**. Michelle 흐름과 비슷한 **shuffle 통제 선별**을 **일 단위**에서 보고 싶을 때 **추가로** 돌리면 됩니다.

**로컬 구조**: `behavior_task/autoresearch/` 안에 두었다면 그 폴더는 **별도 git 저장소**(중첩 clone)입니다. 다른 위치에 받으려면:  
`git clone https://github.com/limserenahansol/autoresearch-behavior.git`

---

## Quick commands / 빠른 실행

```bash
python michelle_style_behavior_triage.py --fast
python michelle_style_neural_behavior_triage.py --demo-synthetic --fast
```

See `README.md` for full flags.  
자세한 옵션은 루트 `README.md`를 참고하세요.
