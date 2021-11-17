import optuna
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss

import pickle

class StackingKfold():
    __X:pd.DataFrame = None
    __y:pd.DataFrame = None
    __X_test:pd.DataFrame = None
    __categorical_columns = None

    def __init__(self, X, y, X_test, categorical_columns):
        self.__X = X
        self.__y = y
        self.__X_test = X_test
        self.__categorical_columns = categorical_columns

    def train_catboost(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = folds.split(self.__X, self.__y)
        cat_val = np.zeros((self.__X.shape[0], 3))
        cat_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_catboost', 'rb') as f:
            params = pickle.load(f)
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]
            train_data = Pool(data=X_train, label=y_train, cat_features=self.__categorical_columns)
            valid_data = Pool(data=X_valid, label=y_valid, cat_features=self.__categorical_columns)

            model = CatBoostClassifier(**params)
            model.fit(train_data, eval_set=valid_data, early_stopping_rounds=5000, verbose=5000, use_best_model=True)

            cat_val[valid_idx] = model.predict_proba(X_valid)
            cat_test += model.predict_proba(self.__X_test) / n_folds

        log_score = log_loss(self.__y, cat_val)
        print(f"Catboost Log Loss Score: {log_score:.5f}\n")
        return cat_val, cat_test

    def train_lgbm(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = folds.split(self.__X, self.__y)
        lgbm_val = np.zeros((self.__X.shape[0], 3))
        lgbm_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_lgbm', 'rb') as f:
            params = pickle.load(f)
        params['learning_rate'] = 0.25
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                         callbacks=[early_stopping(stopping_rounds=5000), log_evaluation(period=5000)])

            lgbm_val[valid_idx] = model.predict_proba(X_valid)
            lgbm_test += model.predict_proba(self.__X_test) / n_folds

        log_score = log_loss(self.__y, lgbm_val)
        print(f"LGBM Log Loss Score: {log_score:.5f}\n")
        return lgbm_val, lgbm_test

    def train_xgb(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = folds.split(self.__X, self.__y)
        xgb_val = np.zeros((self.__X.shape[0], 3))
        xgb_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_xgb', 'rb') as f:
            params = pickle.load(f)
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      early_stopping_rounds=2500, verbose=False, eval_metric='mlogloss')

            xgb_val[valid_idx] = model.predict_proba(X_valid)
            xgb_test += model.predict_proba(self.__X_test) / n_folds

        log_score = log_loss(self.__y, xgb_val)
        print(f"Xgb Log Loss Score: {log_score:.5f}\n")
        return xgb_val, xgb_test

    def train_rf(self, n_folds):
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = folds.split(self.__X, self.__y)
        rf_val = np.zeros((self.__X.shape[0], 3))
        rf_test = np.zeros((self.__X_test.shape[0], 3))

        params = {}
        with open('./data/params/best_param_rf', 'rb') as f:
            params = pickle.load(f)
        print(params)

        for fold, (train_idx, valid_idx) in enumerate(splits):
            X_train, X_valid = self.__X.iloc[train_idx], self.__X.iloc[valid_idx]
            y_train, y_valid = self.__y.iloc[train_idx], self.__y.iloc[valid_idx]

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            rf_val[valid_idx] = model.predict_proba(X_valid)
            rf_test += model.predict_proba(self.__X_test) / n_folds
            print(f"Log Loss Score: {log_loss(y_valid, rf_val[valid_idx]):.5f}")

        log_score = log_loss(self.__y, rf_val)
        print(f"RandomForest Log Loss Score: {log_score:.5f}\n")
        return rf_val, rf_test

    def stakcing(self, n_folds):
        cat_val, cat_test = self.train_catboost(n_folds)
        lgbm_val, lgbm_test = self.train_lgbm(n_folds)
        xgb_val, xgb_test = self.train_xgb(n_folds)
        rf_val, rf_test = self.train_rf(n_folds)

        train_pred = np.concatenate([cat_val, lgbm_val, xgb_val, rf_val], axis=1)
        test_pred = np.concatenate([cat_test, lgbm_test, xgb_test, rf_test], axis=1)
        print(train_pred.shape, test_pred.shape)

        with open('./data/stacking_train', 'wb') as f:
            pickle.dump(train_pred, f)
        with open('./data/stacking_test', 'wb') as f:
            pickle.dump(test_pred, f)