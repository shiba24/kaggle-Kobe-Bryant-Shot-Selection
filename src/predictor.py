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


class Model(object):
    def __init__(self, datadir = "/Users/shintaro/work/kaggle-kobe/data/"):
        self.datadir = datadir

    def makesubmission(self, predict_y, savename="submission99.csv"):
        submit_df = pd.read_csv(self.datadir + "sample_submission.csv")
        submit_df["shot_made_flag"] = predict_y
        submit_df = submit_df.fillna(np.nanmean(predict_y))
        submit_df.to_csv(self.datadir + savename, index=False)

    def xgboost(self, train_X, train_y, test_X, params=None, num_boost_round=32):
        if params is None:
            params = {'objective': 'multi:softprob',
                      'eval_metric': 'mlogloss',
                      'colsample_bytree': 0.55,
                      'min_child_weight': 9.0, 
                      'subsample': 1.0, 
                      'learning_rate': 0.03,
                      'eta': 0.2, 
                      'max_depth': 7.0, 
                      'gamma': 0.75,
                      'num_class': 2,
                      'n_estimators': 580.0
                      }
        dtrain = xgb.DMatrix(train_X, label=train_y)
        dtest = xgb.DMatrix(test_X)
        self.bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        test_y = self.bst.predict(dtest)
        # self.classifier = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=550, subsample=0.5, colsample_bytree=0.5, seed=seed)
        # self.classifier.fit(train_X, train_y)
        # test_y = self.classifier.predict_proba(test_X)[:, 1]
        return test_y


    def randomforest(self, df):
        rfc = RandomForestClassifier(n_estimators=30)
        rfc.fit(train_X, train_y)
        test_y = rfc.predict_proba(test_X)[:, 1]
        return test_y


    def randomsearch_xgboost(df):
        param_distributions={'max_depth': sp.stats.randint(1, 11),
                             'subsample': sp.stats.uniform(0.25, 0.75),
                             'colsample_bytree': sp.stats.uniform(0.25, 0.75)
        }
        xgb_model = XGBClassifier()
        rs = RandomizedSearchCV(xgb_model,
                                param_distributions,
                                cv=10,
                                n_iter=20,
                                scoring="log_loss",
                                n_jobs=1,
                                verbose=2)
        rs.fit(train_X, train_y.transpose()[0]) 
        predict = rs.predict_proba(test_X)
        return predict[:, 1]


if __name__ == "__main__":
    dp = Dataprocess()
    train_X, train_y, test_X = dp.process()
    Model = Model()
    y = Model.xgboost(train_X, train_y, test_X)
    Model.makesubmission(y, savename="submission_n.csv")



"""

    def logistic_distance(train_df, test_df, weight=10.0):
        Logi = LogisticRegression(solver="newton-cg")
        fit_x = np.expand_dims(train_df["shot_distance"].as_matrix(), 1)
        predict_x = np.expand_dims(test_df["shot_distance"].as_matrix(), 1)
        fit_y = np.expand_dims(train_df["shot_made_flag"].as_matrix(), 1)

        weights = np.ones(len(fit_x))
        ind, = np.where(train_df["shot_distance"] > 40)
        weights[ind] = weight
        Logi.fit(fit_x, fit_y, weights)
        predict_y = Logi.predict_proba(predict_x)
        return predict_y[:, 1]


    def logistic_distance_and_locxy(train_df, test_df, weight=10.0):
        Logi = LogisticRegression(solver="newton-cg")
        fit_y = np.expand_dims(train_df["shot_made_flag"].as_matrix(), 1)
        weights = np.ones(len(fit_y))
        ind, = np.where(train_df["shot_distance"] > 40)
        weights[ind] = weight

        fit_x = train_df[["shot_distance", "loc_x", "loc_y"]].as_matrix()
        predict_x = test_df[["shot_distance", "loc_x", "loc_y"]].as_matrix()

        Logi.fit(fit_x, fit_y, weights)
        predict_y = Logi.predict_proba(predict_x)
        return predict_y[:, 1]

"""

