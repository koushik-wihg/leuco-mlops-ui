import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HFSE_REE_Ratios(BaseEstimator, TransformerMixin):
    def __init__(self, candidates=None):
        self.candidates = candidates or [
            ("Nb", "Y"), ("Zr", "Y"), ("Th", "Yb"),
            ("Ce", "Yb"), ("La", "Ce"), ("Nb", "La"),
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdf = X.copy()
        for num, den in self.candidates:
            col_num = next((c for c in Xdf.columns if c.lower() == num.lower()), None)
            col_den = next((c for c in Xdf.columns if c.lower() == den.lower()), None)
            if col_num and col_den:
                num_vals = pd.to_numeric(Xdf[col_num], errors="coerce")
                den_vals = pd.to_numeric(Xdf[col_den], errors="coerce").replace({0: np.nan})
                Xdf[f"{num}_{den}"] = (num_vals / den_vals).fillna(0.0)
        return Xdf


class PivotILRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, comp_cols=(), zero_replace_factor=1e-6):
        self.comp_cols = tuple(comp_cols)
        self.zero_replace_factor = zero_replace_factor

    def fit(self, X, y=None):
        self.input_columns_ = list(X.columns)
        self.comp_cols_ = [c for c in self.comp_cols if c in self.input_columns_]
        self.comp_idx_ = [self.input_columns_.index(c) for c in self.comp_cols_]

        if len(self.comp_idx_) > 0:
            comp_vals = X.iloc[:, self.comp_idx_].to_numpy(dtype=float)
            pos = comp_vals[(~np.isnan(comp_vals)) & (comp_vals > 0)]
            self.eps_ = (pos.min() * self.zero_replace_factor) if pos.size > 0 else self.zero_replace_factor
        else:
            self.eps_ = self.zero_replace_factor

        self.noncomp_cols_ = [c for c in self.input_columns_ if c not in self.comp_cols_]
        return self

    def _close(self, A):
        s = A.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return A / s

    def transform(self, X):
        Xdf = pd.DataFrame(X, columns=self.input_columns_)
        if len(self.comp_cols_) < 2:
            return Xdf.copy()

        comps = Xdf.loc[:, list(self.comp_cols_)].to_numpy(dtype=float)
        comps = np.where(np.isnan(comps), self.eps_, comps)
        comps = np.where(comps <= 0, self.eps_, comps)

        Xc = self._close(comps)
        n, k = Xc.shape
        if k < 2:
            return Xdf.copy()

        ilr = np.zeros((n, k - 1))
        for j in range(k - 1):
            gm = np.exp(np.mean(np.log(Xc[:, j + 1:]), axis=1))
            scale = math.sqrt((k - j - 1) / (k - j))
            ilr[:, j] = scale * np.log(Xc[:, j] / gm)

        ilr_cols = [f"ilr_{c}_vs_rest" for c in self.comp_cols_[:-1]]
        ilr_df = pd.DataFrame(ilr, columns=ilr_cols, index=Xdf.index)
        noncomp_df = Xdf[self.noncomp_cols_]
        return pd.concat([ilr_df, noncomp_df], axis=1)
