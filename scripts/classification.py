import pandas as pd
from amlearn.learn.classifier import AmClassifier
from imblearn.ensemble import EasyEnsemble
from sklearn.ensemble import GradientBoostingClassifier

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"

"""
This is an example script of performing classification with cross-validation
and dataset undersampling, if the dataset is imbalanced. 
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

# classifier setting
classifier_params = \
    {"max_depth": 2, "max_features": 'sqrt', "n_estimators": 300,
     "min_samples_split": .001, "min_samples_leaf": .001, "verbose": 1}

# cross-validation and undersampling (if imbalanced) settings
cv_num = 5
imbalance_num = 3
random_state = 2019
imbalanced_params = {"random_state": random_state, "n_subsets": imbalance_num}

# load data
features_df = pd.read_pickle(features_file)
target_df = pd.read_csv(target_file, index_col=0)

# ML
am_classifier = AmClassifier(output_path=output_path,
                             classifier=GradientBoostingClassifier,
                             classifier_params=classifier_params)

am_classifier.fit(features_df[feature_cols], target_df[target_col],
                  scoring=['roc_auc', 'accuracy', 'f1', 'precision', 'recall'],
                  imblearn=True, imblearn_method=EasyEnsemble,
                  imblearn_params=imbalanced_params, cv_num=cv_num,
                  random_state=random_state)
