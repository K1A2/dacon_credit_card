import catboost.utils
import matplotlib.pyplot as plt
import numpy as np

import DataIO
import Preprocessing
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix
import Model as md
import  ParamTuning
import pickle

def auto_encoder():
    dataio = DataIO.DataReadWrite()
    preprocesser = Preprocessing.PreprocesserGBoost()
    df = dataio.read_csv_to_df('train.csv')
    df_test = dataio.read_csv_to_df('test.csv')

    # 모든 데이터가 중복 되는 열 제거
    # df = df.drop_duplicates(df.columns)

    # 클래스 분리
    y = df.iloc[:, -1]
    df = df.drop(['credit'], axis=1)

    # 데이터 전처리
    X = preprocesser.data_preprocess_2(df, 'train')
    X_submmit = preprocesser.data_preprocess_2(df_test, 'test')

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=2424)
    print(X_train.shape)
    input = Input(shape=(X_train.shape[1],))
    encoded = Dense(128, activation='relu')(input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded2 = Dense(128, activation='relu')(encoded)
    decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded2)
    # autoencoder
    autoencoder = Model(input, decoded)
    # encoder
    encoder = Model(input, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_crossentropy'])
    autoencoder.fit(X_train, X_train, batch_size=128, epochs=100, validation_data=(X_test, X_test), callbacks=[md.AutoSaveCallback()])

def train_catboost(X, y, X_submmit, n_splits):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=756353)
    outcomes = []
    sub = np.zeros((X_submmit.shape[0], 3))
    for n_fold, (train_index, val_index) in enumerate(folds.split(X, y)):
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index, :], y.iloc[val_index, :]

        clf = CatBoostClassifier(iterations=50000, task_type="CPU", objective='MultiClass', thread_count=12, random_seed=444)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5000, verbose=1)

        predictions = clf.predict_proba(X_val)

        logloss = log_loss(to_categorical(y_val), predictions)
        outcomes.append(logloss)
        print(f"FOLD {n_fold} : logloss:{logloss}")

        # d = pd.DataFrame({"importance": clf.get_feature_importance(),
        #                   "label": X.columns.tolist()})
        # sns.barplot(data=d, y='label', x='importance')
        # plt.show()

        sub += clf.predict_proba(X_submmit)

    mean_outcome = np.mean(outcomes)
    my_submission = sub / folds.n_splits
    print("Mean:{}".format(mean_outcome))

    submission = pd.read_csv('./data/sample_submission.csv')
    submission.loc[:, 1:] = my_submission
    submission.to_csv(f'./data/submission/{n_splits}_{mean_outcome}_xgboost.csv', index=False)

'''
{
objective: MultiClass
depth: 9
learning_rate: 0.03
grow_policy: Lossguide
bootstrap_type: Bernoulli
l2_leaf_reg: 2
task_type: GPU
random_state: 1234
subsample: 0.86129349174007
max_leaves: 41
iterations: 80000
}
'''

def only_catbooost():
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

    y = y.loc[X.index,:]
    X = X.reset_index()
    X = X.drop(['index'], axis=1)
    y = y.reset_index()
    y = y.drop(['index'], axis=1)

    print(categorical_columns)
    # for i in range(10, 26):
    #     train_catboost(X, y, X_submmit, i)

    tunner = ParamTuning.Tuner(X, y, categorical_columns)

    # best_param_catboost = tunner.tuning_catboost(n_trials=1)
    # print(best_param_catboost)
    # with open('./data/params/best_param_catboost', 'wb') as f:
    #     pickle.dump(best_param_catboost, f)
    tunner.tuning_lgbm()

# def stacking():
    # def stack_catboost():
    #

def stakcing():
    def get_stacked(model, X_train, y_train, X_test, n_folds):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=False)
        train_fold_pred = np.zeros((X_train.shape[0], 1))
        test_pred = np.zeros((X_test.shape[0], n_folds))
        print(model.__class__.__name__, ' model 시작 ')

        for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train, y=y_train)):
            print('\t 폴드 세트: ', folder_counter, ' 시작 ')
            X_tr = X_train[train_index]
            y_tr = y_train[train_index]
            X_te = X_train[valid_index]

            model.fit(X_tr, y_tr)
            train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1, 1)
            test_pred[:,folder_counter] = model.predict(X_test)
        test_pred_mean = np.mean(test_pred ,axis=1).reshape((-1, 1))
        return train_fold_pred, test_pred_mean

    dataio = DataIO.DataReadWrite()
    preprocesser = Preprocessing.PreprocesserGBoost()
    df = dataio.read_csv_to_df('train.csv')
    df_test = dataio.read_csv_to_df('test.csv')

    # 모든 데이터가 중복 되는 열 제거
    # df = df.drop_duplicates(df.columns)

    # 클래스 분리
    y = df.iloc[:, -1].values
    df = df.drop(['credit'], axis=1)

    # 데이터 전처리
    X = preprocesser.data_preprocess_2(df)
    X_submmit = preprocesser.data_preprocess_2(df_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=2424)

    # CV스태킹 알고리즘 각 모델에 적용
    knn = KNeighborsClassifier(n_neighbors=3)
    rf = RandomForestClassifier()
    adb = AdaBoostClassifier()
    dt = DecisionTreeClassifier()

    knn_train, knn_test = get_stacked(knn, X_train, y_train, X_test, 5)
    rf_train, rf_test = get_stacked(rf, X_train, y_train, X_test, 5)
    dt_train, dt_test = get_stacked(dt, X_train, y_train, X_test, 5)
    ada_train, ada_test = get_stacked(adb, X_train, y_train, X_test, 5)

    # CV스태킹 알고리즘 결과로 메타 모델 학습/시험에 필요한 result_a result_b 만들기
    Stack_final_X_train = np.concatenate((knn_train, rf_train, dt_train, ada_train), axis=1)
    Stack_final_X_test = np.concatenate((knn_test, rf_test, dt_test, ada_test), axis=1)
    print(Stack_final_X_train.shape, Stack_final_X_test.shape)

    model = CatBoostClassifier()
    model.fit(Stack_final_X_train, y_train, eval_set=(Stack_final_X_test, y_test))
    y_pred = model.predict_proba(X_test)
    loss = log_loss(to_categorical(y_test), y_pred)
    print(f"CatBoostClassifier log_loss: {loss}")

    pred_sub = model.predict_proba(X_submmit)
    submission = pd.read_csv('./data/sample_submission.csv')
    submission.loc[:, 1:] = pred_sub
    # submission.to_csv(f'./data/submission/{loss}_stack.csv', index=False)

def main():
    # auto_encoder()
    only_catbooost()
    # stakcing()

if __name__ == '__main__':
    main()