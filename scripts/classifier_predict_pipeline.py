import pandas as pd
from amlearn.learn.classifier import AmClassifier
from amlearn.learn.predict_pipeline import PredictPipeline
from imblearn.ensemble import EasyEnsemble
from sklearn.ensemble import GradientBoostingClassifier

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"

"""
This is an example script of prediction pipeline from a trained classification 
model. 
Please upgrade amlearn to versions later than v0.3.4 to make sure everything
is right.
"""

system = ["Cu65Zr35", "qr_5plus10^10"]

# define file location and information
features_file = "xxx/features_all.pickle.gz"
feature_cols = ["xxx", ..., "xxx"]

target_file = "xxx/target.csv"
target_col = "xxx"

model_file = "xxx/xxx.pkl"
output_path = "xxx/xxx"

# load data
features_df = pd.read_pickle(features_file)
target_df = pd.read_csv(target_file, index_col=0)

# prediction
predictions = load_model_and_predict(
    X=feature_df[feature_cols], y=target_df[target_col],
    model_file=model_file, task='classification', output_path=output_path,
    scoring=['roc_auc', 'accuracy', 'f1', 'precision', 'recall'])
