"""Train, compare and save the application-scoring model.

Run:  python -m src.train
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from . import config as C
from .data import clean, load_raw
from .evaluate import best_threshold, plot_diagnostics, summary_metrics, threshold_report


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), C.NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), C.CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight=None,
            n_jobs=-1, random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=25, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=rs),
    }


def main():
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw())
    features = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
    X, y = df[features], df["Default_Flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    print(f"Rows: {len(df):,} | default rate: {y.mean():.1%} | train {len(X_tr):,} / test {len(X_te):,}")

    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["roc_auc", "average_precision"])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        m = summary_metrics(y_te, p)
        rows.append({"model": name,
                     "cv_auc_mean": scores["test_roc_auc"].mean(),
                     "cv_auc_std": scores["test_roc_auc"].std(),
                     "test_auc": m["roc_auc"], "test_gini": m["gini"],
                     "test_ks": m["ks"], "test_pr_auc": m["pr_auc"]})
        fitted[name] = (pipe, p)
        print(f"{name:<20} CV AUC {rows[-1]['cv_auc_mean']:.3f} +/- {rows[-1]['cv_auc_std']:.3f} | "
              f"test AUC {m['roc_auc']:.3f} KS {m['ks']:.3f}")

    results = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    results.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)

    # Select on cross-validation (not the test set) to avoid optimistic bias.
    best_name = results.iloc[0]["model"]
    best_pipe, p_test = fitted[best_name]

    # Pick the decision threshold on OUT-OF-FOLD training predictions so the
    # hold-out test set stays untouched and the reported results are honest.
    oof = cross_val_predict(best_pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    t, _ = best_threshold(y_tr, oof)
    report = threshold_report(y_te, p_test, t)
    plot_diagnostics(y_te, p_test, best_name, C.FIG_DIR / "model_diagnostics.png")

    imp = permutation_importance(best_pipe, X_te, y_te, scoring="roc_auc",
                                 n_repeats=3, random_state=C.RANDOM_STATE, n_jobs=1)
    imp_df = pd.Series(imp.importances_mean, index=features).sort_values()
    ax = imp_df.plot.barh(figsize=(7, 4), color="#2a6f97")
    ax.set(title=f"Permutation importance ({best_name})", xlabel="Drop in test AUC")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "feature_importance.png", dpi=140)
    plt.close()

    joblib.dump({"pipeline": best_pipe, "threshold": t, "features": features,
                 "model_name": best_name}, C.MODELS_DIR / "credit_model.joblib")
    summary = {"best_model": best_name, "decision": report,
               "importance": imp_df.sort_values(ascending=False).round(4).to_dict()}
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest model: {best_name} | cost-optimal threshold {t:.2f}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
