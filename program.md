# Autoresearch Program: Period Prediction Decoder

## Goal
Maximize **per_mouse_acc** using leave-one-mouse-out cross-validation.
Predict which experimental period (Pre/During/Post/Withdrawal/Re-exposure) each
mouse-day belongs to, using only behavioral and physiological features.

## Current Best
- per_mouse_acc = 0.888 (5-class, RF200, 30 features)
- per_mouse_acc = 0.927 (2-class morphine vs water, RF200)

## Rules
1. Only modify `pipeline.py`
2. `run()` must print `METRIC per_mouse_acc=X.XXXXXX`
3. Keep leave-one-mouse-out CV (14 mice)
4. Must use all 14 mice
5. Complete within 120 seconds
6. ONE change per experiment

## Allowed Modifications
- Feature selection / engineering
- Preprocessing (scaling, imputation, outlier handling)
- Model type and hyperparameters
- Target definition (5-class, 2-class, etc.)
- Ensemble methods

## Domain Knowledge
- 14 mice: 6 Active (self-administer morphine), 8 Passive (yoked)
- Passive mice have NaN for lick/reward during "During" (day 6-10) -- by design
- RequirementLast (PR breakpoint) = most important behavioral metric
- Pharmacological data (TST, HOT) has many NaN (measured on specific days only)
- Transition days (6, 11, 14) are less reliable
- Post and Re-exposure periods show strongest Active vs Passive separation
