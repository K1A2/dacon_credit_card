import optuna
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

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
                    param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

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
        for key, value in param_defulat.items():
            print("    {}: {}".format(key, value))
        if param_defulat['grow_policy'] == 'Lossguide':
            param_defulat['boosting_type'] = 'Plain'
        return param_defulat

    def tuning_lgbm(self):
        categorical_columns = self.__categorical_columns
        train_x = self.__train_x
        train_y = self.__train_y
        valid_x = self.__valid_x
        valid_y = self.__valid_y

        model = LGBMClassifier()
        data = [[0,1,3],[1,2,1],[1,1,3]]
        data = np.array(data)

        y = np.array([1,0,1])
        model.fit(data, y)
        print(model.predict_proba(data))

    def tuning_xgboost(self):
        categorical_columns = self.__categorical_columns
        train_x = self.__train_x
        train_y = self.__train_y
        valid_x = self.__valid_x
        valid_y = self.__valid_y
