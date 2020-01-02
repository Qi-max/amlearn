import pandas as pd
from amlearn.learn.predict import load_model_and_predict
from amlearn.utils.data import read_lammps_dump

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"

"""
This is an example script of predicting on an arbitrary dataset from a trained 
classification model. 
Please upgrade amlearn to versions later than v0.3.4 to make sure everything
is right.
"""

lammps_df = read_lammps_dump("xxx")
target_col = "xxx"

feature_file = "xxx/features_all.pickle.gz"
feature_df = pd.read_pickle(feature_file)
feature_cols = list(feature_df.columns)

model_file = "xxx/model_1.pkl"
output_path = "xxx/xxx"

# predict
predictions = load_model_and_predict(
    model_file=model_file,
    X=feature_df[feature_cols], y=lammps_df[target_col],
    task='classification', output_path=output_path)
