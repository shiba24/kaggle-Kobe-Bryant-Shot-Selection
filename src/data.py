import pandas as pd
import numpy as np
import scipy as sp
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
<<<<<<< HEAD
from sklearn import mixture


class Dataprocess(object):
    datafile = "data.csv"
    def __init__(self, datadir = "/Users/shintaro/work/kaggle-kobe/data/"):
        self.datadir = datadir

    def read(self):
        self.df_orig = pd.read_csv(self.datadir + self.datafile)
        self.df = self.df_orig.copy()

    def process(self):
        self.read()
        self.preproc()
        self.set_mapper()
        self.split_df()
        train_X = self.vec_X(self.train_df)
        train_y = self.vec_y(self.train_df)
        test_X = self.mapper_X.transform(self.test_df)
        return train_X, train_y, test_X


    def preproc(self):
        self.df["time_remaining"] = self.df["minutes_remaining"] * 60 + self.df["seconds_remaining"]
        self.df['last_5_sec'] = self.df['time_remaining'] < 5
        self.df['latter_half'] = self.df['time_remaining'] < 360
        self.df['first_period'] = self.df['period'] == 1
        self.df['latter_period'] = self.df['period'] > 2
        self.df['last_period'] = self.df['period'] == 4
        self.df['last_quarter'] = self.df['time_remaining'] < 180

        threshold = 3
        anomaly = 14
        self.df['last_moment'] = self.df.apply(lambda row: row['time_remaining'] < threshold or row['time_remaining'] == anomaly, axis=1)
        self.df['away'] = self.df.matchup.str.contains('@')
        self.df['secondsFromStart'] = 60 * (11 - self.df['minutes_remaining']) + (60 - self.df['seconds_remaining'])
        self.df['secondsFromGameStart'] = (self.df['period'] <= 4).astype(int) * (self.df['period'] - 1) * 12 * 60 + (self.df['period'] > 4).astype(int) * ((self.df['period'] - 4) * 5 * 60 + 3 * 12 * 60) + self.df['secondsFromStart']
        numGaussians = 13
        gaussianMixtureModel = mixture.GMM(n_components=numGaussians, covariance_type='full', 
                                           params='wmc', init_params='wmc',
                                           random_state=1, n_init=3,  verbose=0)
        gaussianMixtureModel.fit(self.df.ix[:,['loc_x','loc_y']])
        self.df['shotLocationCluster'] = gaussianMixtureModel.predict(self.df.ix[:,['loc_x','loc_y']])
        self.df['homeGame'] = self.df['matchup'].apply(lambda x: 1 if (x.find('@') < 0) else 0)

        self.df["game_year"] = pd.Series([int(self.df["game_date"][i][:4]) for i in range(0, len(self.df))])
        self.df["game_month"] = pd.Series([int(self.df["game_date"][i][5:7]) for i in range(0, len(self.df))])
        self.df["game_day"] = pd.Series([int(self.df["game_date"][i][-2:]) for i in range(0, len(self.df))])

        action_type_list = list(set(self.df["action_type"].tolist()))
        self.df["action_type_num"] = pd.Series([action_type_list.index(self.df["action_type"][i]) for i in range(0, len(self.df))])

        combined_shot_type_list = list(set(self.df["combined_shot_type"].tolist()))
        self.df["combined_shot_type_num"] = pd.Series([combined_shot_type_list.index(self.df["combined_shot_type"][i]) for i in range(0, len(self.df))])

        opponent_list = list(set(self.df["opponent"].tolist()))
        self.df["opponent_num"] = pd.Series([opponent_list.index(self.df["opponent"][i]) for i in range(0, len(self.df))])

        game_id_list = list(set(self.df["game_id"].tolist()))
        self.df["game_id_num"] = pd.Series([game_id_list.index(self.df["game_id"][i]) for i in range(0, len(self.df))])

        season_list = list(set(self.df["season"].tolist()))
        season_list.sort()
        self.df["season_num"] = pd.Series([season_list.index(self.df["season"][i]) for i in range(0, len(self.df))])

        self.df["shot_distance"][self.df["shot_distance"] > 45] = 45

        # del self.df["team_id"], self.df["team_name"], self.df["game_event_id"], self.df["lat"], self.df["lon"]
        # return self.df


    def set_mapper(self):
        self.mapper_X = DataFrameMapper([
            (u'action_type', LabelBinarizer()),
            (u'combined_shot_type', LabelBinarizer()),
            (u'loc_x', None),
            (u'loc_y', None),
            (u'minutes_remaining', None),
            (u'period', LabelBinarizer()),

            (u'playoffs', LabelBinarizer()),
            (u'season', LabelBinarizer()),
            (u'seconds_remaining', None),
            (u'shot_distance', None),
            (u'shot_type', LabelBinarizer()),
            (u'shot_zone_area', LabelBinarizer()),
            (u'shot_zone_basic', LabelBinarizer()),
            (u'shot_zone_range', LabelBinarizer()),
            (u'matchup', LabelBinarizer()),
            (u'shot_id', None),

            (u'season_num', None),
            (u'game_year', None),
            (u'game_month', None),
            (u'game_day', None),

            (u'first_period', LabelBinarizer()),
            (u'latter_period', LabelBinarizer()),
            (u'last_period', LabelBinarizer()),
            (u'last_quarter', LabelBinarizer()),
            (u'time_remaining', None),
            (u'latter_half', LabelBinarizer()),
            (u'last_5_sec', LabelBinarizer()),
            (u'opponent_num', LabelBinarizer()),
            (u'game_id_num', LabelBinarizer()),

            (u'last_moment', LabelBinarizer()),
            (u'away', LabelBinarizer()),
            (u'secondsFromStart', None),
            (u'secondsFromGameStart', None),
            (u'shotLocationCluster', LabelBinarizer()),
            (u'homeGame', LabelBinarizer()),
            ])
        self.mapper_y = DataFrameMapper([(u'shot_made_flag', None),])
        self.mapper_X.fit(self.df)
        self.mapper_y.fit(self.df)


    def split_df(self):
        self.train_df = self.df[~np.isnan(self.df["shot_made_flag"])]
        self.test_df = self.df[np.isnan(self.df["shot_made_flag"])]


    def vec_X(self, df):
        return self.mapper_X.transform(df.copy())


    def vec_y(self, df):
        return self.mapper_y.transform(df.copy())
=======
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
>>>>>>> bf62235bc612fb16e960a65435aa0ef3801cddbf


if __name__ == "__main__":
    dp = Dataprocess()
    a, b, c = dp.process()



"""

if __name__ == "__main__":

    # logistic
    # train_df, test_df = split(df)
    # predict_y = logistic_distance_and_locxy(train_df, test_df)
    # makesubmission(predict_y, savename="submission_logistic.csv")

    # xgboost
    predict_y = xgboost_mappedvec(df, seed=0)
    # predict_2 = xgboost_mappedvec(df, seed=1)
    # predict_3 = xgboost_mappedvec(df, seed=2)
    # predict_4 = xgboost_mappedvec(df, seed=3)
    # predict_5 = xgboost_mappedvec(df, seed=4)
    # predict_6 = xgboost_mappedvec(df, seed=5)
    # predict_7 = xgboost_mappedvec(df, seed=6)
    # predict_8 = xgboost_mappedvec(df, seed=7)
    # predict_y = np.mean((predict_1, predict_2, predict_3, predict_4, predict_5, predict_6, predict_7, predict_8), axis=0)
    # predict_y2 = xgboost_mappedvec(df)    
    # predict_y3 = xgboost_mappedvec(df)    

    makesubmission(predict_y, savename="submission_502.csv")

"""
"""
(u'game_id',
(u'season',
(u'shot_made_flag',
(u'opponent',
(u'game_date',
(u'action_type_num', 
(u'combined_shot_type_num', 


loc = train_df[["loc_x", "loc_y"]].as_matrix()
flag = train_df["shot_made_flag"].as_matrix()


heatmap_all, xedges, yedges = np.histogram2d(loc[:, 0], loc[:, 1], bins=100)

heatmap_success, xedges, yedges = np.histogram2d(loc[flag==1, 0], loc[flag==1, 1], bins=100)


sns.heatmap(heatmap_all, vmax=50)

success_rate = heatmap_success / heatmap_all

success_rate[np.is]




def mean_game(train_df, test_df):
    game_id_list = list(set(df["game_id"].tolist()))
    success_rate_game = np.array([train_df["shot_made_flag"][train_df["game_id"]==game_id_list[i]].mean() for i in range(0, len(game_id_list))])

    predict_y = success_rate_game[test_df["game_id_num"]]
    return predict_y


def mean_all(train_df, test_df):
    return train_df["shot_made_flag"].mean()




namelist = ["action_type", "combined_shot_type", "game_event_id",
            "game_id", "lat", "loc_x", "loc_y", "lon", "minutes_remaining",
            "period", "playoffs", "season", "seconds_remaining", "shot_distance",
            "shot_made_flag", "shot_type", "shot_zone_area", "shot_zone_basic",
            "shot_zone_range", "team_id", "team_name", "game_date", "matchup",
            "opponent", "shot_id"]


y_pred = clf.predict(train_X)

print("Number of mislabeled points out of a total %d points : %d"  % (train_X.shape[0],(train_y != y_pred).sum()))

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    print(ll)
    return ll
    
logloss(train_y,clf.predict_proba(train_X)[:,1])



randomSeed = 1
numFolds = 4
mainLearner = ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=5, 
                                            min_samples_leaf=120, max_features=120, 
                                            criterion='entropy', bootstrap=False, 
                                            n_jobs=-1, random_state=randomSeed)


RandomizedSearchCV(cv=10, error_score='raise',
          estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1),
          fit_params={}, iid=True, n_iter=20, n_jobs=1,
          param_distributions={'subsample': <scipy.stats._distn_infrastructure.rv_frozen object at 0x10d0f2450>, 'colsample_bytree': <scipy.stats._distn_infrastructure.rv_frozen object at 0x10d0f24d0>, 'max_depth': <scipy.stats._distn_infrastructure.rv_frozen object at 0x10d0f2250>},
          pre_dispatch='2*n_jobs', random_state=None, refit=True,
          scoring='log_loss', verbose=2)





"""








"""

def score(params):
    print "Training with params : "
    print params
    N_boost_round=[]
    Score=[]
    skf = cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True, random_state=25)
    for train, test in skf:
        X_Train, X_Test, y_Train, y_Test = X_train[train], X_train[test], y_train[train], y_train[test]
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
    return {'loss': Average_best_score, 'status': STATUS_OK}


def optimize(trials):
    space = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        
        #Control complexity of model
        "eta" : hp.quniform("eta", 0.2, 0.6, 0.05),
        "max_depth" : hp.quniform("max_depth", 1, 10, 1),
        "min_child_weight" : hp.quniform('min_child_weight', 1, 10, 1),
        'gamma' : hp.quniform('gamma', 0, 1, 0.05),
        'learning_rate': hp.quniform('learning_rate', 0., 0.1, 0.01),
        'n_estimators': hp.quniform('n_estimators', 500, 600, 10),
        #Improve noise robustness 
        "subsample" : hp.quniform('subsample', 0.5, 1, 0.05),
        "colsample_bytree" : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        
        'num_class' : 2,
        'silent' : 1}
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)
    print "best parameters",best


df = pd.read_csv(datadir + datafile)
df = preproc(df)

mapper_X, mapper_y = mapper(df)
train_df, test_df = split(df)
X_train = mapper_X.transform(train_df.copy())
y_train = mapper_y.transform(train_df.copy())
y_train = y_train.transpose()[0]


np.random.seed(0)

trials = Trials()
optimize(trials)

#     Average of best iteration 64.5
#     Score 0.6018852
# best parameters {'colsample_bytree': 0.6000000000000001, 'min_child_weight': 7.0, 'subsample': 0.9, 'eta': 0.2, 'max_depth': 6.0, 'gamma': 0.9}
# best parameters {'colsample_bytree': 0.55, 'learning_rate': 0.03, 'min_child_weight': 9.0, 'n_estimators': 580.0, 'subsample': 1.0, 'eta': 0.2, 'max_depth': 7.0, 'gamma': 0.75}

#Adapt best params
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

score(params)

# Training with params : 
# train-mlogloss:0.564660 eval-mlogloss:0.608842
# Average of best iteration 32.0
# Score 0.6000522


X_test = mapper_X.transform(test_df.copy())
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

dtrain = xgb.DMatrix(X_train.as_matrix(),label=y_train.tolist())
dtest = xgb.DMatrix(X_test.as_matrix())
bst = xgb.train(params, dtrain, num_boost_round=32)

pred = bst.predict(dtest)

"""
