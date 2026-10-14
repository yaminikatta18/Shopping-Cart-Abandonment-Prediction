"""Trains Decision Tree + Random Forest, evaluates them, saves models and metrics.json."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from generate_data import generate

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cart_data.csv"
FEATURES = ["browsing_time", "products_viewed", "cart_value",
            "previous_purchases", "discount_usage"]
TARGET = "abandoned"
SEED = 42


def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    step = max(1, len(fpr) // 60)  # thin the curve for the browser
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
    }


def main():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        generate().to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]

    # M5: held-out test set, stratified so the class ratio is preserved
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                criterion="gini", random_state=SEED).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                oob_score=True, n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)

    # M5: stratified 5-fold cross-validation on the training split
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_scores = {n: cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy").tolist()
                 for n, m in [("Decision Tree", dt), ("Random Forest", rf)]}

    # M3: feature importance (gini + permutation for the forest)
    perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=SEED)
    importance = {
        "features": FEATURES,
        "decision_tree": dt.feature_importances_.tolist(),
        "random_forest": rf.feature_importances_.tolist(),
        "rf_permutation": perm.importances_mean.tolist(),
    }

    # M3/M5: validation curve over tree depth (diagnose over-fitting)
    depths = list(range(1, 16))
    curve = {"depths": depths, "train": [], "test": []}
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
        curve["train"].append(t.score(X_tr, y_tr))
        curve["test"].append(t.score(X_te, y_te))

    metrics = {
        "dataset": {"rows": len(df), "train_rows": len(X_tr), "test_rows": len(X_te),
                    "abandon_rate": float(y.mean()), "features": FEATURES,
                    "stats": df[FEATURES].describe().loc[["min", "mean", "max"]].round(2).to_dict()},
        "models": {"Decision Tree": evaluate(dt, X_te, y_te),
                   "Random Forest": evaluate(rf, X_te, y_te)},
        "rf_oob_score": float(rf.oob_score_),
        "cv_accuracy": cv_scores,
        "importance": importance,
        "depth_curve": curve,
        "tree_rules": export_text(dt, feature_names=FEATURES, max_depth=3),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(dt, ROOT / "models" / "decision_tree.joblib")
    joblib.dump(rf, ROOT / "models" / "random_forest.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for name, m in metrics["models"].items():
        print(f"{name:14s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")
    print("Saved models and metrics to /models")


if __name__ == "__main__":
    main()
