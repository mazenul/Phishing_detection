import pandas as pd
import numpy as np
import joblib
import time
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEP = "=" * 60

# ── Load full dataset ─────────────────────────────────────────
print(SEP)
print("  LightGBM — Full Dataset Training")
print(SEP)

df = pd.read_csv("ml_balanced.csv")
X = df.drop(columns=["label"])
y = df["label"]

print(f"\n  Samples  : {len(df):,}")
print(f"  Features : {X.shape[1]}")
print(f"  Class 0  : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  Class 1  : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

# ── Model — hyperparams confirmed best by 5-fold CV ──────────
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ── Train on full dataset ─────────────────────────────────────
print("\n  Training on full dataset ...")
start = time.time()
model.fit(X, y)
elapsed = time.time() - start
print(f"  Done in {elapsed:.1f}s")

# ── In-sample metrics (expected ~99% — confirmed by CV) ───────
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print(f"\n  In-sample metrics (reference only — CV is the true estimate):")
print(f"    Accuracy  : {accuracy_score(y, y_pred):.4f}")
print(f"    Precision : {precision_score(y, y_pred):.4f}")
print(f"    Recall    : {recall_score(y, y_pred):.4f}")
print(f"    F1-Score  : {f1_score(y, y_pred):.4f}")
print(f"    ROC-AUC   : {roc_auc_score(y, y_prob):.4f}")

# ── Top 10 feature importances ────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)
print(f"\n  Top 10 features:")
for rank, (feat, imp) in enumerate(top10.items(), 1):
    print(f"    {rank:>2}. {feat:<35} {imp}")

# ── Save model ────────────────────────────────────────────────
joblib.dump(model, "lightgbm_model.pkl")
print(f"\n  Model saved -> lightgbm_model.pkl")
print(SEP)
