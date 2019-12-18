import pandas as pd
from amlearn.learn.regressor import AmRegressor
from sklearn.ensemble import GradientBoostingRegressor

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"

"""
This is an example script of performing regression with cross-validation. 
Please upgrade amlearn to versions later than v0.3.1 to make sure everything
is right.
"""

system = ["Cu65Zr35", "qr_5plus10^10"]

# define file location and information
features_file = "xxx/features_all.pickle.gz"
feature_cols = ["xxx", ..., "xxx"]

target_file = "xxx/target.csv"
target_col = "xxx"

output_path = "xxx/xxx"

# regressor setting
regressor_params = \
    {"max_depth": 5, "max_features": 'sqrt', "n_estimators": 500,
     "min_samples_split": .001, "min_samples_leaf": .001, "verbose": 1}

# cross-validation setting
cv_num = 5
random_state = 2019

# load data
features_df = pd.read_pickle(features_file)
target_df = pd.read_csv(target_file, index_col=0)

# ML
am_regressor = AmRegressor(output_path=output_path,
                           regressor=GradientBoostingRegressor,
                           regressor_params=regressor_params)

am_regressor.fit(features_df[feature_cols], target_df[target_col],
                 scoring=['r2', 'neg_mean_absolute_error',
                          'neg_mean_squared_error'],
                 cv_num=cv_num, random_state=random_state)
