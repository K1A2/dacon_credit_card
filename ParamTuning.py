import optuna
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss

class Tuner():
    __train_x:pd.DataFrame = None
    __train_y:pd.DataFrame = None
    __valid_x:pd.DataFrame = None
    __valid_y:pd.DataFrame = None
    __categorical_columns = None

    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, categorical_columns):
        self.__train_x, self.__valid_x,self.__train_y, self.__valid_y =\
            train_test_split(X, y, test_size=0.2, stratify=y, random_state=678988)
        self.__categorical_columns = categorical_columns

    def tuning_catboost(self, n_trials):
        categorical_columns = self.__categorical_columns
        train_x = self.__train_x
        train_y = self.__train_y
        valid_x = self.__valid_x
        valid_y = self.__valid_y

        param_defulat = {}
        param_defulat['objective'] = 'MultiClass'
        param_defulat['iterations'] = 80000
        param_defulat['allow_writing_files'] = False
        param_defulat['learning_rate'] = 0.015
        param_defulat['task_type'] = 'CPU'
        param_defulat['random_state'] = 1234
        def objective(trial):
            try:
                param = {
                    "depth": trial.suggest_int("depth", 5, 16),
                    "grow_policy": trial.suggest_categorical(
                        "grow_policy", ['SymmetricTree', 'Lossguide']
                    ),
                    "bootstrap_type": trial.suggest_categorical(
                        "bootstrap_type", ["Bayesian", "Bernoulli"]
                    ),
                    "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 50),
                }

                if param["bootstrap_type"] == "Bayesian":
                    param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
                elif param["bootstrap_type"] == "Bernoulli":
                    param["subsample"] = trial.suggest_float("subsample", 0.5, 1)

                if param['grow_policy'] != "Lossguide":
                    param['boosting_type'] = 'Plain'

                if param['grow_policy'] == "Lossguide":
                    param['max_leaves'] = trial.suggest_int('max_leaves', 16, 64)

                for k, v in param_defulat.items():
                    param[k] = v

                print(f"{trial.number}번\n" + "{")
                for k, v in param.items():
                    print(f"{k}: {v}")
                print("}")

                train_data = Pool(train_x, train_y, cat_features=categorical_columns)
                vaild_data = Pool(valid_x, valid_y, cat_features=categorical_columns)

                clf = CatBoostClassifier(**param)
                clf.fit(train_data, eval_set=vaild_data, early_stopping_rounds=5000, verbose=5000)

                predictions = clf.predict_proba(valid_x)
                logloss = log_loss(to_categorical(valid_y), predictions)

                print(f"{trial.number}번 logloss: {logloss}")

                return logloss
            except:
                return 100

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")

        for key, value in trial.params.items():
            param_defulat[key] = value
        if param_defulat['grow_policy'] == 'Lossguide':
            param_defulat['boosting_type'] = 'Plain'
        for key, value in param_defulat.items():
            print("    {}: {}".format(key, value))
        return param_defulat

    def tuning_lgbm(self, n_trials):
        train_x = self.__train_x.values
        train_y = self.__train_y.values.ravel()
        valid_x = self.__valid_x.values
        valid_y = self.__valid_y.values.ravel()

        param_defulat = {}
        param_defulat['learning_rate'] = 0.015
        param_defulat["objective"] = "multiclass"
        param_defulat['n_jobs'] = 12
        param_defulat['metric'] = 'multi_logloss'
        param_defulat['device_type'] = 'cuda'
        param_defulat['random_state'] = 1234
        param_defulat['n_estimators'] = 80000
        param_defulat['boosting_type'] = 'cuda'

        def objective(trial):
            try:
                param = {
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 20),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_samples': trial.suggest_int('min_data_in_leaf', 5, 1000),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 120.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'max_bin': trial.suggest_int('max_bin', 1, 255),
                }

                param['num_leaves'] = round((2 ** param['max_depth']) * 0.6)

                for k, v in param_defulat.items():
                    param[k] = v

                print(f"{trial.number}번\n" + "{")
                for k, v in param.items():
                    print(f"{k}: {v}")
                print("}")

                lgbm = LGBMClassifier(**param,)
                lgbm.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                         callbacks=[early_stopping(stopping_rounds=5000), log_evaluation(period=5000)])

                predictions = lgbm.predict_proba(valid_x)
                logloss = log_loss(to_categorical(valid_y), predictions)

                print(f"{trial.number}번 logloss: {logloss}")

                return logloss
            except:
                return 100

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")

        for key, value in trial.params.items():
            param_defulat[key] = value
        param_defulat['num_leaves'] = round(2 ** param_defulat['max_depth'] * 0.6)
        for key, value in param_defulat.items():
            print("    {}: {}".format(key, value))
        return param_defulat

    def tuning_xgboost(self, n_trials):
        train_x = self.__train_x.values
        train_y = self.__train_y.values.ravel()
        valid_x = self.__valid_x.values
        valid_y = self.__valid_y.values.ravel()

        train_y = train_y.astype(int)
        valid_y = valid_y.astype(int)

        param_defulat = {}
        param_defulat['learning_rate'] = 0.015
        param_defulat['n_jobs'] = 12
        param_defulat['n_estimators'] = 80000
        param_defulat['random_state'] = 1234
        param_defulat['booster'] = 'gbtree'
        param_defulat['tree_method'] = 'hist'
        param_defulat['objective'] = 'multi:softmax'
        param_defulat['use_label_encoder'] = False

        def objective(trial):
            try:
                param = {
                    'gamma': trial.suggest_float('gamma', 0, 10),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 120.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'grow_policy': trial.suggest_categorical(
                        'grow_policy', ['lossguide']
                    ),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                }

                if param['grow_policy'] == 'lossguide':
                    param['max_leaves'] = round((2 ** param['max_depth']) * 0.8)

                for k, v in param_defulat.items():
                    param[k] = v

                print(f"{trial.number}번\n" + "{")
                for k, v in param.items():
                    print(f"{k}: {v}")
                print("}")

                xgb = XGBClassifier(**param,)
                xgb.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                        early_stopping_rounds=2500, verbose=False, eval_metric='mlogloss')

                predictions = xgb.predict_proba(valid_x)
                logloss = log_loss(to_categorical(valid_y), predictions)

                print(f"{trial.number}번 logloss: {logloss}")

                return logloss
            except:
                return 100

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")

        for key, value in trial.params.items():
            param_defulat[key] = value
        if param_defulat['grow_policy'] == 'lossguide':
            param_defulat['max_leaves'] = round(2 ** param_defulat['max_depth'] * 0.8)
        for key, value in param_defulat.items():
            print("    {}: {}".format(key, value))
        return param_defulat

    def tuning_rf(self):
        train_x = self.__train_x.values
        train_y = self.__train_y.values.ravel()
        valid_x = self.__valid_x.values
        valid_y = self.__valid_y.values.ravel()

        train_y = train_y.astype(int)
        valid_y = valid_y.astype(int)

        param_defulat = {
            'n_estimators': [10, 100, 300, 500],
            'max_depth': [6, 8, 10, 12, 16, 24],
            'min_samples_leaf': [8, 12, 18, 22],
            'min_samples_split': [8, 16, 20, 24],
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=12)
        grid_cv = GridSearchCV(rf, param_grid=param_defulat, cv=3, n_jobs=12)
        grid_cv.fit(train_x, train_y)

        print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
        print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

        param_defulat = {
            'random_state': 42,
            'n_jobs': 12
        }
        for k, v in grid_cv.best_params_.items():
            param_defulat[k] = v

        rf = RandomForestClassifier(**param_defulat)
        rf.fit(train_x, train_y)
        predictions = rf.predict_proba(valid_x)
        logloss = log_loss(to_categorical(valid_y), predictions)
        print(f"RandomForest logloss: {logloss}")

        return param_defulat