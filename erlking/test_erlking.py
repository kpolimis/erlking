import pandas as pd
import numpy as np
import numpy.testing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.cross_validation as xval
from erlking import random_forest_error, calc_inbag, _core_computation

def test_core_computation():
    inbag_ex = np.array([[ 1.,  2.,  0.,  1.],
                         [ 1.,  0.,  2.,  0.],
                         [ 1.,  1.,  1.,  2.]])

    columns = ['y','x1','x2']
    X_train_ex = np.array([[3,3],[6,4],[6,6]])
    X_test_ex = np.array([[5,2],[5,5]])
    pred_centered_ex = np.array([[-20,-20,10,30],[-20,30,-20,10]])
    n_trees = 4

    our_vij = _core_computation(X_train_ex, X_test_ex, inbag_ex, pred_centered_ex, n_trees)
    r_vij = np.array([ 112.5,  387.5])

    npt.assert_almost_equal(our_vij, r_vij)

#def test_random_forest_error():
    # data = np.array([[70,5,2],[100,5,5],[60,3,3],[100,6,4],[120,6,6]])
    # X = pd.DataFrame(data=data, columns=columns)
    # y_test_ex = [100,70]
    # y_train_ex = [60,100,12]

    # y = X['y']
    # forest = RandomForestRegressor(n_estimators=n_trees)
    # forest.fit(X_train_ex, y_train_ex)

#    y_hat, V_IJ_unbiased = random_forest_error(forest, inbag_ex, X_train_ex, X_test_ex)
