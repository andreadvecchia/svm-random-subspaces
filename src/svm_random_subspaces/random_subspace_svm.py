from __future__ import annotations

from typing import List, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

class RandomSubspaceSVM(BaseEstimator, ClassifierMixin):
    """Ensemble of SVMs trained on random feature subsets.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Typically `LinearSVC()` or `SVC(kernel="rbf")`.
    M : int
        Number of base models (subspaces).
    k : int
        Subspace dimension (number of features per model).
    voting : {'hard','soft'}
        'hard' uses majority vote; 'soft' averages decision_function.
    random_state : int or numpy RandomState, optional
        Reproducibility.
    scale : bool
        If True, includes a StandardScaler before the SVM in each base learner.
    """
    def __init__(self, base_estimator=None, M: int = 25, k: int = 20, voting: str = 'soft',
                 random_state: Optional[int] = None, scale: bool = True):
        self.base_estimator = base_estimator
        self.M = int(M)
        self.k = int(k)
        self.voting = voting
        self.random_state = random_state
        self.scale = scale

    def fit(self, X, y):
        if self.base_estimator is None:
            # Late import to avoid a hard dependency unless used
            from sklearn.svm import LinearSVC
            self.base_estimator = LinearSVC(dual=False)

        rs = check_random_state(self.random_state)
        n_features = X.shape[1]
        if self.k <= 0 or self.k > n_features:
            raise ValueError(f"k must be in [1, {n_features}] (got {self.k}).")

        self.subspaces_: List[np.ndarray] = []
        self.models_: List[Pipeline] = []

        for _ in range(self.M):
            feat_idx = rs.choice(n_features, size=self.k, replace=False)
            self.subspaces_.append(np.sort(feat_idx))

            est = clone(self.base_estimator)
            steps = []
            if self.scale:
                steps.append(('scaler', StandardScaler(with_mean=True)))
            steps.append(('clf', est))
            pipe = Pipeline(steps)
            pipe.fit(X[:, feat_idx], y)
            self.models_.append(pipe)
        return self

    def decision_function(self, X):
        if not hasattr(self, 'models_'):
            raise RuntimeError("Model not fitted.")
        scores = None
        for pipe, idx in zip(self.models_, self.subspaces_):
            clf = pipe.named_steps['clf']
            if hasattr(clf, 'decision_function'):
                s = pipe.decision_function(X[:, idx])
            elif hasattr(clf, 'predict_proba'):
                proba = pipe.predict_proba(X[:, idx])
                s = proba[:, 1] - proba[:, 0]
            else:
                preds = pipe.predict(X[:, idx])
                s = (preds * 2 - 1).astype(float)
            scores = s if scores is None else scores + s
        return scores / self.M

    def predict(self, X):
        if self.voting == 'soft':
            return (self.decision_function(X) >= 0).astype(int)
        else:
            votes = None
            for pipe, idx in zip(self.models_, self.subspaces_):
                pred = pipe.predict(X[:, idx])
                votes = pred if votes is None else votes + pred
            return (votes >= (self.M / 2)).astype(int)
