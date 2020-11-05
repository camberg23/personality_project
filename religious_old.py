import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=23)

# Average CV score on the training set was: 0.03316013008095471
exported_pipeline = make_pipeline(
    PCA(iterated_power=3, svd_solver="randomized"),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.7500000000000001, min_samples_leaf=13, min_samples_split=17, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=8, min_samples_split=18, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 23)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
