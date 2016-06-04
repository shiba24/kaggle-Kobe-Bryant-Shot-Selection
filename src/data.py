import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

from xgboost.sklearn import XGBClassifier
import xgboost as xgb


datadir = "/Users/shintaro/work/kaggle-kobe/data/"      # please change here. this dir includes data.csv and sample_submission.csv
datafile = "data.csv"


def mapper(df):
    x_mapper = DataFrameMapper([
        (u'action_type', LabelBinarizer()),
        (u'combined_shot_type', LabelBinarizer()),
        (u'loc_x', None),
        (u'loc_y', None),
        (u'minutes_remaining', None),
        (u'period', LabelBinarizer()),
        (u'playoffs', LabelBinarizer()),
        (u'seconds_remaining', None),
        (u'shot_distance', None),
        (u'shot_type', LabelBinarizer()),
        (u'shot_zone_area', LabelBinarizer()),
        (u'shot_zone_basic', LabelBinarizer()),
        (u'shot_zone_range', LabelBinarizer()),
        (u'matchup', LabelBinarizer()),
        (u'shot_id', None),
        (u'time_remaining', None),
        (u'opponent_num', LabelBinarizer()),
        (u'game_id_num', LabelBinarizer()),
        ])
    x_mapper.fit(df)
    y_mapper = DataFrameMapper([
        (u'shot_made_flag', None),
        ])
    y_mapper.fit(df)
    return x_mapper, y_mapper


def xgboost_mappedvec(df):
    x_mapper, y_mapper = mapper(df)
    train_df, test_df = split(df)
    train_x_vec = x_mapper.transform(train_df.copy())
    train_y_vec = y_mapper.transform(train_df.copy())
    test_x_vec = x_mapper.transform(test_df.copy())

    clf = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=550, subsample=0.5, colsample_bytree=0.5, seed=0)

    clf.fit(train_x_vec, train_y_vec)
    test_y_vec = clf.predict_proba(test_x_vec)[:, 1]
    return test_y_vec


def preproc(df):
    df["time_remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]

    action_type_list = list(set(df["action_type"].tolist()))
    df["action_type_num"] = pd.Series([action_type_list.index(df["action_type"][i]) for i in range(0, len(df))])

    combined_shot_type_list = list(set(df["combined_shot_type"].tolist()))
    df["combined_shot_type_num"] = pd.Series([combined_shot_type_list.index(df["combined_shot_type"][i]) for i in range(0, len(df))])

    opponent_list = list(set(df["opponent"].tolist()))
    df["opponent_num"] = pd.Series([opponent_list.index(df["opponent"][i]) for i in range(0, len(df))])

    game_id_list = list(set(df["game_id"].tolist()))
    df["game_id_num"] = pd.Series([game_id_list.index(df["game_id"][i]) for i in range(0, len(df))])

    del df["team_id"], df["team_name"], df["game_event_id"], df["lat"], df["lon"]
    return df


def split(df):
    train_df = df[~np.isnan(df["shot_made_flag"])]
    test_df = df[np.isnan(df["shot_made_flag"])]
    return train_df, test_df


def makesubmission(predict_y, savename="submission99.csv"):
    submit_df = pd.read_csv(datadir + "sample_submission.csv")
    submit_df["shot_made_flag"] = predict_y
    submit_df = submit_df.fillna(np.nanmean(predict_y))
    submit_df.to_csv(savename, index=False)


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


def mean_game(train_df, test_df):
    game_id_list = list(set(df["game_id"].tolist()))
    success_rate_game = np.array([train_df["shot_made_flag"][train_df["game_id"]==game_id_list[i]].mean() for i in range(0, len(game_id_list))])

    predict_y = success_rate_game[test_df["game_id_num"]]
    return predict_y


def mean_all(train_df, test_df):
    return train_df["shot_made_flag"].mean()


if __name__ == "__main__":
    df = pd.read_csv(datadir + datafile)
    df = preproc(df)

    # logistic
    train_df, test_df = split(df)
    predict_y = logistic_distance_and_locxy(train_df, test_df)
    makesubmission(predict_y, savename="submission_logistic.csv")

    # xgboost
    predict_y = xgboost_mappedvec(df)    
    makesubmission(predict_y, savename="submission_xgboost.csv")



"""
(u'game_id', (u'season',
(u'shot_made_flag', (u'opponent', (u'game_date',
(u'action_type_num', 
(u'combined_shot_type_num', 



loc = train_df[["loc_x", "loc_y"]].as_matrix()
flag = train_df["shot_made_flag"].as_matrix()


heatmap_all, xedges, yedges = np.histogram2d(loc[:, 0], loc[:, 1], bins=100)

heatmap_success, xedges, yedges = np.histogram2d(loc[flag==1, 0], loc[flag==1, 1], bins=100)


sns.heatmap(heatmap_all, vmax=50)

success_rate = heatmap_success / heatmap_all

success_rate[np.is]




namelist = ["action_type", "combined_shot_type", "game_event_id",
            "game_id", "lat", "loc_x", "loc_y", "lon", "minutes_remaining",
            "period", "playoffs", "season", "seconds_remaining", "shot_distance",
            "shot_made_flag", "shot_type", "shot_zone_area", "shot_zone_basic",
            "shot_zone_range", "team_id", "team_name", "game_date", "matchup",
            "opponent", "shot_id"]


y_pred = clf.predict(train_x)

print("Number of mislabeled points out of a total %d points : %d"  % (train_x.shape[0],(train_y != y_pred).sum()))

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    print(ll)
    return ll
    
logloss(train_y,clf.predict_proba(train_x)[:,1])

"""
