import numpy as np
from sklearn.ensemble.forest import _generate_sample_indices

def calc_inbag(n_samples, forest):
    """

    """
    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    for t_idx in range(n_trees):
        sample_idx.append(_generate_sample_indices(forest.estimators_[t_idx].random_state,
                                                   n_samples))
        inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    return inbag

def _core_computation(X_train, X_test, inbag, pred_centered, n_trees):
    cov_hat = np.zeros((X_train.shape[0], X_test.shape[0]))

    for t_idx in range(n_trees):
        inbag_r = (inbag[:, t_idx] - 1).reshape(-1, 1)
        pred_c_r = pred_centered.T[t_idx].reshape(1, -1)
        cov_hat += np.dot(inbag_r, pred_c_r) / n_trees
    V_IJ = np.sum(cov_hat ** 2, 0)
    return V_IJ

def _bias_correction(V_IJ, inbag, pred_centered, n_trees):
    n_train_samples = inbag.shape[0]
    n_var =  np.mean(np.square(inbag[0:n_trees]).mean(axis=1).T.view(dtype=np.float64)
    - np.square(inbag[0:n_trees].mean(axis=1)).T.view(dtype=np.float64))
    boot_var = np.square(pred_centered).sum(axis=1)/n_trees
    bias_correction = n_train_samples * n_var * boot_var/n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased

def random_forest_error(forest, inbag, X_train, X_test):
    """
    forest : RandomForest{Regressor, Classifier}

    inbag : ndarray
        The inbag matrix (see `calc_inbag`) for the data with which this was fit

    X : ndarray
        with shape (n_sample, n_features).
    """
    pred = np.array([tree.predict(X_test) for tree in forest]).T
    pred_mean = np.mean(pred, 0)
    pred_centered = pred - pred_mean
    n_trees = forest.n_estimators
    n_train_samples = inbag.shape[0]
    V_IJ = _core_computation(X_train, X_test, inbag, pred_centered, n_trees)
    V_IJ_unbiased = _bias_correction(V_IJ, inbag, pred_centered, n_trees)
    return pred_mean, V_IJ_unbiased
