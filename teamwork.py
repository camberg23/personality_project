import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=232)

# Average CV score on the training set was: 0.06070622237425061
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.001),
    RobustScaler(),
    PCA(iterated_power=7, svd_solver="randomized"),
    VarianceThreshold(threshold=0.001),
    LassoLarsCV(normalize=True)
)
# LassoLarsCV(VarianceThreshold(PCA(RobustScaler(VarianceThreshold(input_matrix, threshold=0.001)), 
# iterated_power=7, svd_solver=randomized), threshold=0.001), normalize=True)

# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 232)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
