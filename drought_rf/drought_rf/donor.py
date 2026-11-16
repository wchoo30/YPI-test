"""Donor-gauge regionalization (Heldmyer et al. 2026, Sect. 2.4), for places with no gauge,
such as ctwater farm cells.

1. Keep at-site models with kappa > 0.4.
2. PCA on their variable-importance vectors; keep PCs explaining 95 % of variance.
3. Regress each PC score on standardized static basin attributes (linear regression).
4. For a target place, predict its PC scores from its attributes and rank donors by the
   variance-weighted absolute difference in PC space.
5. Run the top donors' models on the target's own predictors, average the probabilities,
   and threshold (0.35 in the paper, chosen by maximizing kappa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class DonorSelector:
    def __init__(self, var_explained: float = 0.95, kappa_min: float = 0.4):
        self.var_explained = var_explained
        self.kappa_min = kappa_min

    def fit(self, importances: pd.DataFrame, static: pd.DataFrame, kappa: pd.Series):
        """importances: site x feature; static: site x attribute; kappa: per site."""
        keep = kappa[kappa > self.kappa_min].index.intersection(importances.index).intersection(static.index)
        if len(keep) < 3:
            raise ValueError(f"only {len(keep)} sites pass kappa > {self.kappa_min}; need at least 3")
        imp = importances.loc[keep].fillna(0)
        self.imp_scaler_ = StandardScaler().fit(imp)
        self.pca_ = PCA(n_components=self.var_explained, svd_solver="full").fit(self.imp_scaler_.transform(imp))
        self.scores_ = pd.DataFrame(self.pca_.transform(self.imp_scaler_.transform(imp)), index=keep)
        self.static_cols_ = list(static.columns)
        self.st_scaler_ = StandardScaler().fit(static.loc[keep])
        self.reg_ = LinearRegression().fit(self.st_scaler_.transform(static.loc[keep]), self.scores_.values)
        self.weights_ = self.pca_.explained_variance_ratio_
        return self

    def donors(self, target_static: pd.Series, n: int | None = None, frac: float = 0.01) -> pd.Series:
        z = self.st_scaler_.transform(target_static[self.static_cols_].to_frame().T)
        pc = self.reg_.predict(z)[0]
        dist = (np.abs(self.scores_.values - pc) * self.weights_).sum(axis=1)
        d = pd.Series(dist, index=self.scores_.index).sort_values()
        n = n or max(1, int(round(frac * len(d))))
        return d.iloc[:n]


def donor_predict(models: dict, donor_sites, X_target: pd.DataFrame) -> np.ndarray:
    return np.mean([models[s].predict_proba(X_target) for s in donor_sites], axis=0)
