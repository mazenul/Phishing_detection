import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS = 5
CSV_PATH = "ml_balanced.csv"  # Update with your actual path to the dataset

# ---------------------------------------------------------------------------
# Load & shuffle
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X = df.drop(columns=["label"]).values
y = df["label"].values

print(f"Dataset shape : {df.shape}")
print(f"Class distribution — 0: {(y==0).sum()}  1: {(y==1).sum()}")
print(f"Stratified {N_FOLDS}-Fold Cross-Validation\n")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1
    ),
}

METRICS = ["precision", "recall", "f1", "roc_auc"]

# ---------------------------------------------------------------------------
# K-Fold loop
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_results = {}

for model_name, model in models.items():
    print("=" * 65)
    print(f"  {model_name}")
    print("=" * 65)

    fold_scores = {m: [] for m in METRICS}
    fold_scores["train_accuracy"] = []
    fold_scores["test_accuracy"]  = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        # Train predictions
        y_train_pred = model.predict(X_train)
        train_acc    = accuracy_score(y_train, y_train_pred)

        # Test (validation) predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        test_acc = accuracy_score(y_val, y_pred)

        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        f1   = f1_score(y_val, y_pred, zero_division=0)
        auc  = roc_auc_score(y_val, y_prob)
        cm   = confusion_matrix(y_val, y_pred)

        fold_scores["train_accuracy"].append(train_acc)
        fold_scores["test_accuracy"].append(test_acc)
        fold_scores["precision"].append(prec)
        fold_scores["recall"].append(rec)
        fold_scores["f1"].append(f1)
        fold_scores["roc_auc"].append(auc)

        print(f"  Fold {fold_idx}:")
        print(f"    Train Accuracy : {train_acc:.4f}  |  Test Accuracy : {test_acc:.4f}")
        print(f"    Precision      : {prec:.4f}  |  Recall        : {rec:.4f}")
        print(f"    F1-Score       : {f1:.4f}  |  ROC-AUC       : {auc:.4f}")
        print(f"    Confusion Matrix (Test):\n      TN={cm[0,0]}  FP={cm[0,1]}\n      FN={cm[1,0]}  TP={cm[1,1]}")
        print()

    # Summary for this model
    all_metrics = ["train_accuracy", "test_accuracy"] + METRICS
    print(f"  {'Metric':<16}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*60}")
    for metric in all_metrics:
        vals = fold_scores[metric]
        print(
            f"  {metric:<16}  "
            f"{np.mean(vals):>8.4f}  "
            f"{np.std(vals):>8.4f}  "
            f"{np.min(vals):>8.4f}  "
            f"{np.max(vals):>8.4f}"
        )
    print()

    all_results[model_name] = fold_scores

# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------
print("=" * 75)
print("  FINAL COMPARISON  (mean ± std over 5 folds)")
print("=" * 75)
all_metrics = ["train_accuracy", "test_accuracy"] + METRICS
header = f"{'Model':<16}" + "".join(f"{m:>16}" for m in all_metrics)
print(header)
print("-" * (16 + 16 * len(all_metrics)))
for model_name, scores in all_results.items():
    row = f"{model_name:<16}"
    for m in all_metrics:
        mean = np.mean(scores[m])
        std  = np.std(scores[m])
        row += f"  {mean:.4f}±{std:.4f}"
    print(row)
print()
