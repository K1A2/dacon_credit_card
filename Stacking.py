import optuna
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from xgboost import XGBClassifier, DMatrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss
import copy
import pickle

class StackingKfold():
    __X:pd.DataFrame = None
    __y:pd.DataFrame = None
    __X_test:pd.DataFrame = None
    __categorical_columns = None

    __X_cat:pd.DataFrame = None
    __y_cat:pd.DataFrame = None
    __X_test_cat:pd.DataFrame = None
    __categorical_columns_cat = None

    # Stacking.StackingKfold(train_data, train_data_y, test_data, cate_col)
    def __init__(self, X, y, X_test, categorical_columns):
        self.__X = X.drop(['Owner'], axis=1)
        self.__y = y
        self.__X_test = X_test.drop(['Owner'], axis=1)
        self.__categorical_columns = copy.deepcopy(categorical_columns)
        self.__categorical_columns.remove('Owner')

        self.__X_cat = X.drop(['Owner_r'], axis=1)
        self.__y_cat = y
        self.__X_test_cat = X_test.drop(['Owner_r'], axis=1)
        self.__categorical_columns_cat = copy.deepcopy(categorical_columns)
        self.__categorical_columns_cat.remove('Owner_r')

    def train_catboost(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
        splits = folds.split(self.__X_cat, self.__y_cat)
        cat_val = np.zeros((self.__X_cat.shape[0], 3))
        cat_test = np.zeros((self.__X_test_cat.shape[0], 3))

        params = {}
        with open('./data/params/best_param_catboost', 'rb') as f:
            params = pickle.load(f)
        params['task_type'] = 'GPU'
        # params['class_weights'] = {0: 2.7371198,  1: 1.40721238, 2: 0.51974305}
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"-----------Catboost {fold}번-----------")
            X_train, X_valid = self.__X_cat.iloc[train_idx], self.__X_cat.iloc[valid_idx]
            y_train, y_valid = self.__y_cat.iloc[train_idx], self.__y_cat.iloc[valid_idx]
            train_data = Pool(data=X_train, label=y_train, cat_features=self.__categorical_columns_cat)
            valid_data = Pool(data=X_valid, label=y_valid, cat_features=self.__categorical_columns_cat)

            # model = CatBoostClassifier(**params)
            model = CatBoostClassifier(n_estimators=10000)
            # model.fit(train_data, eval_set=valid_data, early_stopping_rounds=5000, verbose=5000, use_best_model=True)
            model.fit(train_data, eval_set=valid_data, early_stopping_rounds=100, verbose=50, use_best_model=True)

            cat_val[valid_idx] = model.predict_proba(X_valid)
            cat_test += model.predict_proba(self.__X_test_cat) / n_folds

        log_score = log_loss(self.__y_cat, cat_val)
        print(f"Catboost Log Loss Score: {log_score:.5f}\n")
        return cat_val, cat_test

    def train_lgbm(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
        splits = folds.split(self.__X, self.__y)
        lgbm_val = np.zeros((self.__X.shape[0], 3))
        lgbm_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_lgbm', 'rb') as f:
            params = pickle.load(f)
        params['learning_rate'] = 0.25
        # params['class_weight'] = {0: 2.7371198,  1: 1.40721238, 2: 0.51974305}
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"-----------LGBM {fold}번-----------")
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                         callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=50)])

            lgbm_val[valid_idx] = model.predict_proba(X_valid)
            lgbm_test += model.predict_proba(self.__X_test) / n_folds

        log_score = log_loss(self.__y, lgbm_val)
        print(f"LGBM Log Loss Score: {log_score:.5f}\n")
        return lgbm_val, lgbm_test

    def train_xgb(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
        splits = folds.split(self.__X, self.__y)
        xgb_val = np.zeros((self.__X.shape[0], 3))
        xgb_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_xgb', 'rb') as f:
            params = pickle.load(f)
        print(params)

        classes_weights = [2.7371198, 1.40721238, 0.51974305]

        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"-----------XGB {fold}번-----------")
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            weights = np.ones(y_train.shape[0], dtype='float')
            for i, val in enumerate(y_train):
                weights[i] = classes_weights[val]

            model = XGBClassifier(**params)
            train_data = DMatrix(X_train, y_train, enable_categorical=True)
            test_data = DMatrix(X_valid, y_valid, enable_categorical=True)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      early_stopping_rounds=100, verbose=True, eval_metric='mlogloss')

            xgb_val[valid_idx] = model.predict_proba(X_valid)
            xgb_test += model.predict_proba(self.__X_test) / n_folds

        log_score = log_loss(self.__y, xgb_val)
        print(f"Xgb Log Loss Score: {log_score:.5f}\n")
        return xgb_val, xgb_test

    def train_rf(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
        splits = folds.split(self.__X, self.__y)
        rf_val = np.zeros((self.__X.shape[0], 3))
        rf_test = np.zeros((self.__X_test.shape[0], 3))

        classes_weights = [2.7371198, 1.40721238, 0.51974305]

        params = {}
        with open('./data/params/best_param_rf', 'rb') as f:
            params = pickle.load(f)
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"-----------RandomForest {fold}번-----------")
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            weights = np.ones(y_train.shape[0], dtype='float')
            for i, val in enumerate(y_train):
                weights[i] = classes_weights[val]

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            rf_val[valid_idx] = model.predict_proba(X_valid)
            rf_test += model.predict_proba(self.__X_test) / n_folds
            print(f"Log Loss Score: {log_loss(y_valid, rf_val[valid_idx]):.5f}")

        log_score = log_loss(self.__y, rf_val)
        print(f"RandomForest Log Loss Score: {log_score:.5f}\n")
        return rf_val, rf_test

    def train_ada(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
        splits = folds.split(self.__X, self.__y)
        ada_val = np.zeros((self.__X.shape[0], 3))
        ada_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_ada', 'rb') as f:
            params = pickle.load(f)
        print(params)
        params['base_estimator'] = DecisionTreeClassifier(max_depth=params['max_depth'])
        del params['max_depth']

        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"-----------AdaBoost {fold}번-----------")
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            model = AdaBoostClassifier(**params)
            model.fit(X_train, y_train)

            ada_val[valid_idx] = model.predict_proba(X_valid)
            ada_test += model.predict_proba(self.__X_test) / n_folds
            print(f"Log Loss Score: {log_loss(y_valid, ada_val[valid_idx]):.5f}")

        log_score = log_loss(self.__y, ada_val)
        print(f"RandomForest Log Loss Score: {log_score:.5f}\n")
        return ada_val, ada_test

    def stakcing(self, n_folds):
        # ada_val, ada_test = self.train_ada(n_folds)
        cat_val, cat_test = self.train_catboost(n_folds)
        xgb_val, xgb_test = self.train_xgb(n_folds)
        rf_val, rf_test = self.train_rf(n_folds)
        lgbm_val, lgbm_test = self.train_lgbm(n_folds)

        train_pred = np.concatenate([cat_val, lgbm_val, xgb_val, rf_val], axis=1)
        test_pred = np.concatenate([cat_test, lgbm_test, xgb_test, rf_test], axis=1)
        print(train_pred.shape, test_pred.shape)
        with open('./data/stacking_train', 'wb') as f:
            pickle.dump(train_pred, f)
        with open('./data/stacking_test', 'wb') as f:
            pickle.dump(test_pred, f)

    def stacking_last(self, n_folds):
        stacking_train = None
        stacking_test = None
        with open('./data/stacking_train', 'rb') as f:
            stacking_train = pickle.load(f)
        with open('./data/stacking_test', 'rb') as f:
            stacking_test = pickle.load(f)

        print(stacking_train.shape)
        print(stacking_test.shape)

        params = {}
        with open('./data/params/best_param_lgbm_stacked', 'rb') as f:
            params = pickle.load(f)
        print(params)

        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
        stack_val = np.zeros((stacking_train.shape[0], 3))
        stack_test = np.zeros((stacking_test.shape[0], 3))
        for fold, (train_idx, valid_idx) in enumerate(folds.split(stacking_train, self.__y), 1):
            print(f"-----------Stacked LGBM {fold}번-----------")
            X_train, X_valid = stacking_train[train_idx], stacking_train[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)])

            stack_val[valid_idx, :] = model.predict_proba(stacking_train[valid_idx])
            stack_test += model.predict_proba(stacking_test) / n_folds

        loss = log_loss(self.__y, stack_val)
        print(f'{loss}')
        submission = pd.read_csv('./data/sample_submission.csv')
        submission.loc[:, 1:] = stack_test
        submission.to_csv(f'./data/submission/stacking_{n_folds}_{loss}_xgboost.csv', index=False)

        # import Tocken
        # from dacon_submit_api import dacon_submit_api
        # submission.to_csv(f'./data/submission/stacking_xgboost.csv', index=False)
        # result = dacon_submit_api.post_submission_file(
        #     './data/submission/stacking_xgboost.csv',
        #     Tocken.token,
        #     '235832',
        #     'K1A2',
        #     f'stack_{loss}_{n_folds}')

    def tuning_lgbm_stack(self, n_trials, n_folds):
        stacking_train = None
        stacking_test = None
        with open('./data/stacking_train', 'rb') as f:
            stacking_train = pickle.load(f)
        with open('./data/stacking_test', 'rb') as f:
            stacking_test = pickle.load(f)

        param_defulat = {}
        param_defulat['learning_rate'] = 0.015
        param_defulat["objective"] = "multiclass"
        param_defulat['n_jobs'] = 12
        param_defulat['metric'] = 'multi_logloss'
        param_defulat['device_type'] = 'cuda'
        param_defulat['random_state'] = 1234
        param_defulat['n_estimators'] = 100000
        param_defulat['boosting_type'] = 'gbdt'

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

                folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4558)
                stack_val = np.zeros((stacking_train.shape[0], 3))
                stack_test = np.zeros((stacking_test.shape[0], 3))
                for fold, (train_idx, valid_idx) in enumerate(folds.split(stacking_train, self.__y), 1):
                    X_train, X_valid = stacking_train[train_idx], stacking_train[valid_idx]
                    y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

                    model = LGBMClassifier(**param)
                    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                              callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=1500)])

                    stack_val[valid_idx, :] = model.predict_proba(stacking_train[valid_idx])
                    stack_test += model.predict_proba(stacking_test) / n_folds

                loss = log_loss(self.__y, stack_val)

                return loss
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

    def r(self, n_folds, n_trials):
        param_defulat = {}
        with open('./data/params/best_param_catboost', 'rb') as f:
            param_defulat = pickle.load(f)
        def objective(trial):
            try:
                seed = trial.suggest_int('seed', 0, 10000)
                print(trial.number,'번 seed', seed)
                folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = folds.split(self.__X, self.__y)
                cat_val = np.zeros((self.__X.shape[0], 3))
                cat_test = np.zeros((self.__X_test.shape[0], 3))

                for fold, (train_idx, valid_idx) in enumerate(splits):
                    X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
                    y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]
                    train_data = Pool(data=X_train, label=y_train, cat_features=self.__categorical_columns)
                    valid_data = Pool(data=X_valid, label=y_valid, cat_features=self.__categorical_columns)

                    model = CatBoostClassifier(**param_defulat)
                    model.fit(train_data, eval_set=valid_data, early_stopping_rounds=5000, verbose=5000,
                              use_best_model=True)

                    cat_val[valid_idx] = model.predict_proba(X_valid)
                    cat_test += model.predict_proba(self.__X_test) / n_folds

                log_score = log_loss(self.__y, cat_val)
                print(f"Catboost Log Loss Score: {log_score:.5f}\n")

                return log_score
            except:
                return 100

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")

        return trial.params