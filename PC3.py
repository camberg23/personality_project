import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=41)

# Average CV score on the training set was: 0.20244789499402877
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_pipeline(
            SelectPercentile(score_func=f_regression, percentile=90),
            Binarizer(threshold=0.9500000000000001)
        )
    ),
    FastICA(tol=0.8),
    RBFSampler(gamma=0.9500000000000001),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=9, min_samples_split=9, n_estimators=100)),
    RidgeCV()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 41)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
