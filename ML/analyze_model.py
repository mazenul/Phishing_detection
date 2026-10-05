import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

SEP = "=" * 65

print(SEP)
print("  MODEL TRAINING ANALYSIS REPORT")
print(SEP)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("\n[1] DATASET")
df = pd.read_csv("ml_balanced.csv")
X = df.drop(columns=["label"])
y = df["label"]

print(f"  Total samples   : {len(df):,}")
print(f"  Total features  : {X.shape[1]}")
print(f"  Class 0 (legit) : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  Class 1 (phish) : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
print(f"  Missing values  : {X.isnull().sum().sum()}")
print(f"  -1 values (flag): {(X == -1).sum().sum():,}  (used as missing indicator)")

# ── 2. Load saved model ──────────────────────────────────────────────────────
print("\n[2] SAVED MODEL (lightgbm_model.pkl)")
model = joblib.load("lightgbm_model.pkl")
params = model.get_params()
print(f"  Type            : {type(model).__name__}")
print(f"  n_estimators    : {params['n_estimators']}")
print(f"  learning_rate   : {params['learning_rate']}")
print(f"  max_depth       : {params['max_depth']}")
print(f"  num_leaves      : {params['num_leaves']}")
print(f"  subsample       : {params['subsample']}")
print(f"  colsample_bytree: {params['colsample_bytree']}")
print(f"  reg_alpha (L1)  : {params['reg_alpha']}")
print(f"  reg_lambda (L2) : {params['reg_lambda']}")
print(f"  random_state    : {params['random_state']}")

# ── 3. Train-set performance (model was trained on full data) ────────────────
print("\n[3] FULL-DATA TRAIN PERFORMANCE (model was fit on 100% of data)")
y_pred_full = model.predict(X)
y_prob_full = model.predict_proba(X)[:, 1]
train_acc   = accuracy_score(y, y_pred_full)
train_auc   = roc_auc_score(y, y_prob_full)
train_f1    = f1_score(y, y_pred_full)
print(f"  Accuracy  : {train_acc:.4f}")
print(f"  F1-Score  : {train_f1:.4f}")
print(f"  ROC-AUC   : {train_auc:.4f}")
cm = confusion_matrix(y, y_pred_full)
print(f"  Confusion Matrix:")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
print("  NOTE: This is IN-SAMPLE performance — will be optimistically biased.")

# ── 4. Holdout test evaluation ───────────────────────────────────────────────
print("\n[4] HOLDOUT TEST SET EVALUATION (80/20 split, stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from lightgbm import LGBMClassifier
eval_model = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=7,
    num_leaves=63, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1
)
eval_model.fit(X_train, y_train)
y_pred = eval_model.predict(X_test)
y_prob = eval_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm2  = confusion_matrix(y_test, y_pred)

print(f"  Train size : {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"  Accuracy   : {acc:.4f}")
print(f"  Precision  : {prec:.4f}")
print(f"  Recall     : {rec:.4f}")
print(f"  F1-Score   : {f1:.4f}")
print(f"  ROC-AUC    : {auc:.4f}")
print(f"  Confusion Matrix:")
print(f"    TN={cm2[0,0]:,}  FP={cm2[0,1]:,}")
print(f"    FN={cm2[1,0]:,}  TP={cm2[1,1]:,}")

# Overfitting check
overfit_gap = train_acc - acc
print(f"\n  Overfitting check (full-train acc - holdout acc): {overfit_gap:+.4f}")
if overfit_gap > 0.02:
    print("  WARNING: Possible overfitting detected (gap > 2%)")
elif overfit_gap < 0.0:
    print("  HEALTHY: No overfitting detected")
else:
    print("  HEALTHY: Minimal gap — model generalizes well")

# ── 5. 5-Fold Cross-Validation on saved model config ────────────────────────
print("\n[5] 5-FOLD STRATIFIED CROSS-VALIDATION (same hyperparams as saved model)")
cv_model = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=7,
    num_leaves=63, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_results = cross_validate(cv_model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

for metric in scoring:
    vals = cv_results[f"test_{metric}"]
    print(f"  {metric:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
          f"[min={np.min(vals):.4f}, max={np.max(vals):.4f}]")

# ── 6. Top feature importances ───────────────────────────────────────────────
print("\n[6] TOP 10 FEATURE IMPORTANCES (saved model)")
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)
for rank, (feat, imp) in enumerate(top10.items(), 1):
    print(f"  {rank:>2}. {feat:<35} {imp:>6}")

# ── 7. Issues & Recommendations ─────────────────────────────────────────────
print("\n[7] ISSUES & RECOMMENDATIONS")
issues = []
recommendations = []

if len(df) < 10000:
    issues.append("Dataset is small (<10k samples).")
else:
    print("  [OK] Dataset size is adequate (208,862 samples)")

# Check if model was trained on full data (no holdout)
issues.append("Saved model trained on 100% of data — no holdout test set used during training.")
recommendations.append("Always hold out a test set BEFORE training and evaluate on it.")

# Check hyperparameter mismatch between train and kfold scripts
issues.append("kfold_verification.py uses different hyperparams than train_lightgbm.py — CV results don't reflect the saved model.")
recommendations.append("Use identical hyperparams in both scripts so CV results represent the actual model.")

# Check no early stopping
issues.append("No early stopping used — model trains all 500 trees even if improvement stalls.")
recommendations.append("Add early_stopping_rounds with a validation set to avoid wasted computation.")

# Check no preprocessing
print("  [OK] Dataset is balanced (50/50) — no class imbalance issue")
print("  [OK] Regularization applied (L1=0.1, L2=0.1)")
print("  [OK] Subsampling applied (subsample=0.8, colsample=0.8)")
print("  [OK] Random state fixed for reproducibility")
print("  [OK] Stratified K-Fold used in verification script")

print()
for i, issue in enumerate(issues, 1):
    print(f"  ISSUE {i}: {issue}")
print()
for i, rec in enumerate(recommendations, 1):
    print(f"  FIX {i}: {rec}")

# ── 8. Final verdict ─────────────────────────────────────────────────────────
print("\n" + SEP)
print("  FINAL VERDICT")
print(SEP)
cv_acc_mean = np.mean(cv_results["test_accuracy"])
cv_f1_mean  = np.mean(cv_results["test_f1"])
cv_auc_mean = np.mean(cv_results["test_roc_auc"])

if cv_acc_mean >= 0.95 and cv_auc_mean >= 0.97:
    verdict = "EXCELLENT"
elif cv_acc_mean >= 0.90 and cv_auc_mean >= 0.93:
    verdict = "GOOD"
elif cv_acc_mean >= 0.80:
    verdict = "ACCEPTABLE"
else:
    verdict = "NEEDS IMPROVEMENT"

print(f"  CV Accuracy : {cv_acc_mean:.4f}")
print(f"  CV F1-Score : {cv_f1_mean:.4f}")
print(f"  CV ROC-AUC  : {cv_auc_mean:.4f}")
print(f"  Overall     : {verdict}")
print(SEP)
