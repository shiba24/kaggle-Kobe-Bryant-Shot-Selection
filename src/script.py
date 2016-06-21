import pandas as pd
import numpy as np
import scipy as sp

from sklearn.grid_search import RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

from data import Dataprocess
from tuning import Tuner

dp = Dataprocess()
train_X, train_y, test_X = dp.process()

# Model = Model()
# test_y = Model.xgboost(train_X, train_y, test_X)
# Model.makesubmission(y, savename="submission_n.csv")

Tuner = Tuner()
params, results = Tuner.tune(train_X, train_y, test_X, max_evals=2500)


np.random.seed(0)
from predictor import Model
Model = Model()
test_y = Model.xgboost(train_X, train_y, test_X, params=params, num_boost_round=30)

Model.makesubmission(test_y, savename="submission_000.csv")

# plus

all_X = dp.mapper_X.transform(dp.df)
dall = xgb.DMatrix(all_X)

all_y = Model.xgboost(Tuner.train_X, Tuner.train_y, all_X, params=params, num_boost_round=30)
plus_X = np.concatenate([all_X, np.expand_dims(all_y, 1)], axis=1)

train_X_plus = plus_X[~np.isnan(dp.df["shot_made_flag"].as_matrix()), :]
test_X_plus = plus_X[np.isnan(dp.df["shot_made_flag"].as_matrix()), :]
Tuner_plus = tuning.Tuner()

# stop here

params_plus, results_plus = Tuner_plus.tune(train_X_plus, train_y, test_X_plus, max_evals=2500)

# stop here
test_y_plus = Model.xgboost(Tuner_plus.train_X, Tuner_plus.train_y, Tuner_plus.test_X, params=params_plus, num_boost_round=32)
Model.makesubmission(test_y_plus, savename="submission_000_plus.csv")


