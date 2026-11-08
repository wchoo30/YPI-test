"""Model zoo. Every model exposes fit(X, y) and predict_proba(X) -> P(drought)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PAPER_CLASS_WEIGHT = {0: 0.2, 1: 0.8}


class _Wrapped:
    def __init__(self, est):
        self.est = make_pipeline(SimpleImputer(strategy="median"), est)

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.est.predict_proba(X[self.columns_])
        classes = list(self.est.classes_)
        return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    @property
    def feature_importances_(self):
        last = self.est[-1]
        imp = getattr(last, "feature_importances_", None)
        if imp is None and hasattr(last, "coef_"):
            imp = np.abs(last.coef_[0])
        return pd.Series(imp, index=self.columns_).sort_values(ascending=False)


def random_forest(n_estimators: int = 500, min_samples_leaf: int = 10, seed: int = 0, **kw):
    """The paper's model: RF classifier, class weights 0.8/0.2, node size 10, no tuning.

    Gini (mean decrease in impurity) importance is what sklearn reports by default,
    matching the paper's choice.
    """
    return _Wrapped(RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features="sqrt", class_weight=PAPER_CLASS_WEIGHT,
        n_jobs=-1, random_state=seed, **kw))


def lightgbm(seed: int = 0, **kw):
    """Gradient-boosted trees, the model family ctwater already uses."""
    from lightgbm import LGBMClassifier
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                  class_weight=PAPER_CLASS_WEIGHT, random_state=seed, verbose=-1)
    params.update(kw)
    return _Wrapped(LGBMClassifier(**params))


def logistic(**kw):
    """Linear benchmark on the same predictors (scaled)."""
    lr = LogisticRegression(max_iter=2000, class_weight=PAPER_CLASS_WEIGHT, C=0.1, **kw)
    w = _Wrapped(lr)
    w.est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr)
    return w


class Climatology:
    """P(drought) = training-period drought frequency for that site and day-of-year window."""

    def fit(self, X, y):
        doy = X["target_date"].dt.dayofyear if "target_date" in X else X["date"].dt.dayofyear
        df = pd.DataFrame({"doy": doy.values, "y": np.asarray(y)})
        rate = df.groupby("doy")["y"].mean().reindex(range(1, 367))
        # smooth over a 31-day circular window
        r = pd.concat([rate.iloc[-15:], rate, rate.iloc[:15]])
        self.rate_ = r.rolling(31, center=True, min_periods=1).mean().iloc[15:-15].fillna(df["y"].mean())
        return self

    def predict_proba(self, X):
        doy = X["target_date"].dt.dayofyear if "target_date" in X else X["date"].dt.dayofyear
        return self.rate_.reindex(doy.values).values


class Persistence:
    """Gauged sites only: P(drought at t+lead) = 1 if flow is in drought today."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return (X["qpct_0d"].values < 0.20).astype(float)


MODELS = {"rf": random_forest, "lgbm": lightgbm, "logistic": logistic,
          "climatology": Climatology, "persistence": Persistence}
