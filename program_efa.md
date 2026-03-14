# Autoresearch Program: Addiction Index (EFA / PCA)

## Goal
Maximize **quality_score** = 0.50×stability + 0.30×var_explained + 0.20×interpretability.

The addiction index should:
1. Be **stable** (split-half correlation > 0.7)
2. Capture meaningful **variance** (>50% total variance explained)
3. Be **interpretable** (many features with |loading| > 0.4)
4. **Separate** Active vs Passive groups (low p-value on at least one factor)

## Current Best
- quality_score = ??? (baseline not yet established)

## Rules
1. Only modify `pipeline_efa.py`
2. `run()` must print `METRIC quality_score=X.XXXXXX`
3. Must use all 14 mice
4. Complete within 60 seconds
5. ONE change per experiment

## Allowed Modifications
- Feature selection (which behavioral/pupil/pharma variables)
- Delta pairs (which phase comparisons: Post-Pre, Reexp-Pre, etc.)
- Scaling method (StandardScaler, RobustScaler, MinMaxScaler, etc.)
- Number of factors (1, 2, 3, 4)
- Rotation method (varimax, promax, oblimin, none)
- Method (PCA vs sklearn FactorAnalysis)
- NaN threshold for column removal
- Imputation strategy
- Feature transformations (log, z-score, winsorize)

## Domain Knowledge
- 14 mice: 6 Active, 8 Passive
- Passive mice have NaN for lick/reward during "During" → avoid During deltas for Passive
- Delta scores (Post-Pre, Reexp-Pre) capture morphine-induced change
- RequirementLast, lick metrics = core addiction-relevant behaviors
- Pharmacological data (TI, TST, HOT) captures physical dependence
- Post and Re-exposure show strongest behavioral signatures
- Withdrawal captures dependence/craving
