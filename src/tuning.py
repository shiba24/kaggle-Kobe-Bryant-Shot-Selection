import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from sklearn.grid_search import RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn import mixture

from sklearn import datasets, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.cross_validation import KFold
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class Tuner(object):
    """docstring for Ptune"""
    def tune(self, train_X, train_y, test_X, max_evals=2500):
        self.train_X = train_X
        self.train_y = train_y.reshape(len(train_y),)
        self.test_X = test_X
        np.random.seed(0)
        trials = Trials()
        params = self.optimize(trials, max_evals=max_evals)

        #     Average of best iteration 64.5
        #     Score 0.6018852
        # best parameters {'colsample_bytree': 0.6000000000000001, 'min_child_weight': 7.0, 'subsample': 0.9, 'eta': 0.2, 'max_depth': 6.0, 'gamma': 0.9}

        # best parameters {'colsample_bytree': 0.55, 'learning_rate': 0.03,
        #                  'min_child_weight': 9.0, 'n_estimators': 580.0,
        #                  'subsample': 1.0, 'eta': 0.2, 'max_depth': 7.0, 'gamma': 0.75}
        # best params : 2
        #                 {'colsample_bytree': 0.45, 'eta': 0.2,
        #                  'gamma': 0.9500000000000001, 'learning_rate': 0.04,
        #                  'max_depth': 6.0, 'min_child_weight': 9.0,
        #                  'n_estimators': 750.0, 'subsample': 1.84}


        # Adapt best params
        # params = {'objective': 'multi:softprob',
        #           'eval_metric': 'mlogloss',
        #           'colsample_bytree': 0.55,
        #           'min_child_weight': 9.0, 
        #           'subsample': 1.0, 
        #           'learning_rate': 0.03,
        #           'eta': 0.2, 
        #           'max_depth': 7.0, 
        #           'gamma': 0.75,
        #           'num_class': 2,
        #           'n_estimators': 580.0
        #           }


        params_result = self.score(params)

        # Training with params : 
        # train-mlogloss:0.564660 eval-mlogloss:0.608842
        # Average of best iteration 32.0
        # Score 0.6000522
        return params, params_result


    def tuned_predict(self, params, num_boost_round, test_X=None):
        df_train_X = pd.DataFrame(self.train_X)
        if test_X is None:
            df_test_X = pd.DataFrame(self.test_X)
        else:
            df_test_X = pd.DataFrame(test_X)
        dtrain = xgb.DMatrix(df_train_X.as_matrix(), label=self.train_y.tolist())
        dtest = xgb.DMatrix(df_test_X.as_matrix())
        self.bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        pred = self.bst.predict(dtest)
        return pred


    def score(self, params):
        print "Training with params : "
        print params
        N_boost_round=[]
        Score=[]
        skf = cross_validation.StratifiedKFold(self.train_y, n_folds=6, shuffle=True, random_state=25)
        for train, test in skf:
            X_Train, X_Test, y_Train, y_Test = self.train_X[train], self.train_X[test], self.train_y[train], self.train_y[test]
            dtrain = xgb.DMatrix(X_Train, label=y_Train)
            dvalid = xgb.DMatrix(X_Test, label=y_Test)
            watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
            model = xgb.train(params, dtrain, num_boost_round=150, evals=watchlist, early_stopping_rounds=10)
            predictions = model.predict(dvalid)
            N = model.best_iteration
            N_boost_round.append(N)
            score = model.best_score
            Score.append(score)
        Average_best_num_boost_round = np.average(N_boost_round)
        Average_best_score = np.average(Score)
        print "\tAverage of best iteration {0}\n".format(Average_best_num_boost_round)
        print "\tScore {0}\n\n".format(Average_best_score)
        return {'loss': Average_best_score, 'status': STATUS_OK, 'Average_best_num_boost_round': Average_best_num_boost_round}


    def optimize(self, trials, max_evals=250):
        self.space = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            
            #Control complexity of model
            "eta" : hp.quniform("eta", 0.1, 0.3, 0.025),
            "max_depth" : hp.quniform("max_depth", 5, 10, 1),
            "min_child_weight" : hp.quniform('min_child_weight', 5, 10, 1),
            'gamma' : hp.quniform('gamma', 0, 1, 0.05),
            'learning_rate': hp.quniform('learning_rate', 0., 0.1, 0.01),
            'n_estimators': hp.quniform('n_estimators', 500, 800, 10),
            #Improve noise robustness 
            "subsample" : hp.quniform('subsample', 1.0, 2, 0.01),
            "colsample_bytree" : hp.quniform('colsample_bytree', 0.3, 0.6, 0.025),
            
            'num_class' : 2,
            'silent' : 1}
        best = fmin(self.score, self.space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
        print "best parameters", best
        return best

