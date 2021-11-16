import numpy as np
import DataIO
import Preprocessing
import pandas as pd
from catboost import CatBoostClassifier, Pool
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

def train_stacking(X, y, X_submmit, params, n_split):
    pass

def train_catboost(X, y, X_submmit, params, n_splits, categorical):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=756353)
    outcomes = []
    sub = np.zeros((X_submmit.shape[0], 3))
    for n_fold, (train_index, val_index) in enumerate(folds.split(X, y)):
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index, :], y.iloc[val_index, :]

        train_data = Pool(X_train, y_train, cat_features=categorical)
        test_data = Pool(X_val, y_val, cat_features=categorical)

        clf = CatBoostClassifier(**params)
        clf.fit(train_data, eval_set=test_data, early_stopping_rounds=5000, verbose=1000)

        predictions = clf.predict_proba(X_val)

        logloss = log_loss(to_categorical(y_val), predictions)
        outcomes.append(logloss)
        print(f"FOLD {n_fold} : logloss:{logloss}")

        sub += clf.predict_proba(X_submmit)

    mean_outcome = np.mean(outcomes)
    my_submission = sub / folds.n_splits
    print("Mean:{}".format(mean_outcome))

    submission = pd.read_csv('./data/sample_submission.csv')
    submission.loc[:, 1:] = my_submission
    submission.to_csv(f'./data/submission/kfold_5/{n_splits}_{mean_outcome}_xgboost.csv', index=False)

def train_catboost_one(X, y, X_submmit, params):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=678988)

    clf = CatBoostClassifier(**params)
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5000)

    predictions = clf.predict_proba(X_val)
    logloss = log_loss(to_categorical(y_val), predictions)
    print(f"logloss: {logloss}")

    predictions = clf.predict_proba(X_submmit)
    submission = pd.read_csv('./data/sample_submission.csv')
    submission.loc[:, 1:] = predictions
    submission.to_csv(f'./data/submission/one_{logloss}_xgboost.csv', index=False)

def main():
    dataio = DataIO.DataReadWrite()
    preprocesser = Preprocessing.PreprocesserGBoost()
    df = dataio.read_csv_to_df('train.csv')
    df_test = dataio.read_csv_to_df('test.csv')

    # 모든 데이터가 중복 되는 열 제거
    # df = df.drop_duplicates(df.columns)

    # 클래스 분리
    y = df.loc[:, ['credit']]
    # df = df.drop(['credit'], axis=1)

    # 데이터 전처리
    X, numerical_columns, categorical_columns = preprocesser.data_preprocess_2_comb(df.drop_duplicates(), 'train')
    X_submmit, numerical_submmit, categorical_submmit = preprocesser.data_preprocess_2_comb(df_test, 'test')

    y = y.loc[X.index, :]
    X = X.reset_index()
    X = X.drop(['index'], axis=1)
    y = y.reset_index()
    y = y.drop(['index'], axis=1)



    # param = {
    #     'task_type': 'CPU',
    #     'random_seed': 1234,
    #     'thread_count': 12,
    #     'iterations': 40000,
    # }


    # 1번
    # one_0.7056862464826119_xgboost.csv
    # param = {
    #     'objective': 'MultiClass',
    #     'depth': 14,
    #     'learning_rate': 0.06259226791856165,
    #     'grow_policy': 'Lossguide',
    #     'bootstrap_type': 'Bayesian',
    #     'l2_leaf_reg': 22,
    #     'task_type': 'CPU',
    #     'random_seed': 1234,
    #     'thread_count': 12,
    #     'bagging_temperature': 1.5476071404273228,
    #     'max_leaves': 62,
    #     'iterations': 40000,
    # }
    # param 2
    # two_0.7027083784935696_xgboost.csv
    # param = {
    #     'objective': 'MultiClass',
    #     'depth': 10,
    #     'learning_rate': 0.06576219655285793,
    #     'grow_policy': 'Lossguide',
    #     'bootstrap_type': 'Bernoulli',
    #     'l2_leaf_reg': 5,
    #     'task_type': 'CPU',
    #     'random_seed': 1234,
    #     'thread_count': 12,
    #     'subsample': 0.692762589836913,
    #     'max_leaves': 56,
    #     'iterations': 40000,
    # }
    # param 4
    # param = {
    #     'objective': 'MultiClass',
    #     'depth': 9,
    #     'learning_rate': 0.03,
    #     'grow_policy': 'Lossguide',
    #     'bootstrap_type': 'Bernoulli',
    #     'l2_leaf_reg': 2,
    #     'task_type': 'GPU',
    #     'random_state': 1234,
    #     'subsample': 0.86129349174007,
    #     'max_leaves': 41,
    #     'iterations': 80000,
    # }
    #prarm 5
    param = {
        'objective': 'MultiClass',
        'depth': 14,
        'learning_rate': 0.015,
        'grow_policy': 'Lossguide',
        'bootstrap_type': 'Bernoulli',
        'l2_leaf_reg': 3,
        'task_type': 'GPU',
        'random_state': 1234,
        'subsample': 0.8500607447093872,
        'max_leaves': 51,
        'iterations': 80000,
        'allow_writing_files': False
    }

    # Mean Encoding
    # param 3
    # param = {
    #     'objective': 'MultiClass',
    #     'depth': 13,
    #     'learning_rate': 0.013196422959834992,
    #     'grow_policy': 'SymmetricTree',
    #     'bootstrap_type': 'Bayesian',
    #     'l2_leaf_reg': 38,
    #     'task_type': 'GPU',
    #     'random_seed': 1234,
    #     'thread_count': 1024,
    #     'bagging_temperature': 0.09182176591772706,
    #     'boosting_type': 'Plain',
    #     'iterations': 40000,
    # }


    # train_catboost_one(X, y, X_submmit, param)
    print(len(X.columns.tolist()))
    print(len(categorical_columns))
    print(len(numerical_columns))
    for i in range(5, 20):
        print(f"KFold: {i}")
        train_catboost(X, y, X_submmit, param, i, categorical_columns)

if __name__ == '__main__':
    main()