import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from svm_random_subspaces import RandomSubspaceSVM
from sklearn.svm import LinearSVC

def test_fit_predict_shapes():
    X, y = make_classification(n_samples=500, n_features=30, n_informative=8, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = RandomSubspaceSVM(base_estimator=LinearSVC(dual=False), M=10, k=10, random_state=0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    assert y_pred.shape == y_te.shape

def test_deterministic_with_seed():
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6, random_state=42)
    clf1 = RandomSubspaceSVM(M=5, k=8, random_state=123)
    clf2 = RandomSubspaceSVM(M=5, k=8, random_state=123)
    clf1.fit(X, y); clf2.fit(X, y)
    np.testing.assert_allclose(clf1.predict(X), clf2.predict(X))
