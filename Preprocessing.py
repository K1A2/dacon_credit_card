from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

# gender random_under {'F', 'M'}
# income_type 5 {'Commercial associate', 'Student', 'State servant', 'Pensioner', 'Working'}
# Education 5 {'Lower secondary', 'Academic degree', 'Higher education', 'Incomplete higher', 'Secondary / secondary special'}
# family_type 5 {'Married', 'Separated', 'Single / not married', 'Widow', 'Civil marriage'}
# house_type 6 {'Office apartment', 'With parents', 'House / apartment', 'Rented apartment', 'Municipal apartment', 'Co-op apartment'}
# FLAG_MOBIL none {none}
# work_phone random_under {0, none}
# phone random_under {0, none}
# email random_under {0, none}
# occyp_type 19 {'Security staff', 'Laborers', 'HR staff', 'Cleaning staff', 'Waiters/barmen staff', 'Medicine staff', 'Managers', 'Accountants', 'IT staff', 'Realty agents', 'Low-skill Laborers', 'Private service staff', 'none', 'Core staff', 'Drivers', 'Secretaries', 'Sales staff', 'Cooking staff', 'High skill tech staff'}
# car_reality under_tome {0, none, random_under}
# credit under_tome {0.0, none.0, random_under.0}

class Preprocess():
    __default_path = './data/'
    __label_all_data = {'gender': ['F', 'M'],
                        'income_type': ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'],
                        'Education': ['Secondary / secondary special', 'Higher education',
                                      'Incomplete higher', 'Lower secondary', 'Academic degree'],
                        'family_type': ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
                        'house_type': ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment',
                                       'Office apartment', 'Co-op apartment'],
                        'occyp_type': ['none', 'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers',
                                       'High skill tech staff', 'Accountants', 'Medicine staff', 'Cooking staff',
                                       'Security staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers',
                                       'Waiters/barmen staff', 'Secretaries', 'Realty agents', 'HR staff', 'IT staff'],
                        'work_phone': [0, 1], 'phone': [0, 1], 'email': [0, 1], 'car_reality': [0, 1, 2], 'credit': [0.0, 1.0, 2.0]}

    def __z_score_nomalizer(self, type, df:pd.DataFrame):
        data = [df.mean(), df.std()]
        if type == 'test':
            with open(self.__default_path + 'z_score', 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.__default_path + 'z_score', 'wb') as f:
                pickle.dump(data, f)
        return (df - data[0]) / data[1]

    def make_analyze_plot(self, df:pd.DataFrame):
        drop_df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'credit']
        drop_df = drop_df[numeric_column + categorical_column]

        # q1 = drop_df[numeric_column].quantile(0.25)
        # q3 = drop_df[numeric_column].quantile(0.75)
        # iqr = q3 - q1
        # c = (drop_df[numeric_column] < (q1 - 1.5 * iqr)) | (drop_df[numeric_column] > (q3 + 1.5 * iqr))
        # c = c.any(axis=1)
        # drop_df = drop_df.drop(drop_df[c].index, axis=0)
        # print(drop_df)
        #
        # print(drop_df.isnull().sum())

        # for column in categorical_column:
        #     if column not in ['credit']:
        #         datas = dict()
        #         for d in drop_df[[column, 'credit']].values:
        #             if d[0] in datas:
        #                 datas[d[0]][int(d[1])] += 1
        #             else:
        #                 datas[d[0]] = [0,0,0]
        #                 datas[d[0]][int(d[1])] = 1
        #         plt.clf()
        #         sns.heatmap(pd.DataFrame(datas), cmap=sns.light_palette("gray", as_cmap=True), annot=True, fmt="d")
        #         plt.title(f"{column}_heatmap")
        #         plt.savefig(f'./data/analyze/heatmap/{column}_heatmap.png')
        #
        # fig, ax = plt.subplots(1, 4, figsize=(16,4))
        # ax[0].boxplot(drop_df[numeric_column[0]].values)
        # ax[0].set_title(numeric_column[0])
        # ax[1].boxplot(drop_df[numeric_column[1]].values)
        # ax[1].set_title(numeric_column[1])
        # ax[2].boxplot(drop_df[numeric_column[2]].values)
        # ax[2].set_title(numeric_column[2])
        # ax[3].boxplot(drop_df[numeric_column[3]].values)
        # ax[3].set_title(numeric_column[3])
        # plt.show()

        # for i in range(len(categorical_column) - 1):
        #     sns.countplot(data=drop_df, x=categorical_column[i])
        #     plt.title(f"{categorical_column[i]}_count")
        #     plt.savefig(f'./data/analyze/{categorical_column[i]}_count.png')
        # print('f')
        #
        # ppdd = drop_df[numeric_column + ['credit']]
        # sns.pairplot(ppdd, vars=ppdd.columns[:-1], hue='credit')
        # plt.show()
        print('maek plot finish')

    def train_data_preprocess(self, df:pd.DataFrame):
        drop_df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                                 'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'credit']
        drop_df[numeric_column] = self.__z_score_nomalizer('train', drop_df[numeric_column])

        X = drop_df[numeric_column].values

        for column in categorical_column:
            if column not in ['work_phone', 'phone', 'email', 'car_reality', 'credit']:
                encoder = LabelEncoder()
                encoder.fit(drop_df[column].values)
                drop_df[column] = pd.DataFrame(encoder.transform(drop_df[column].values).reshape(-1, 1), columns=[column])
            one_hot_encoder = OneHotEncoder()
            one_hot_encoder.fit(drop_df[column].values.reshape(-1, 1))
            label = one_hot_encoder.transform(drop_df[column].values.reshape(-1, 1)).toarray()
            X = np.concatenate([X, label], axis=1)

        Y = X[:, -3:X.shape[1]]
        X = X[:, :-3]

        resemple = TomekLinks()
        X, Y = resemple.fit_resample(X, Y)

        check = [0,0,0]
        for i in Y.tolist():
            check[i.index(1)] += 1
        print(check)

        print('train preprocessing complete')
        return X, Y

    def train_data_oversampling(self, df:pd.DataFrame):
        drop_df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'credit']
        drop_df = drop_df[numeric_column + categorical_column]

        # 이상치 제거
        q1 = drop_df[numeric_column].quantile(0.25)
        q3 = drop_df[numeric_column].quantile(0.75)
        iqr = q3 - q1
        c = (drop_df[numeric_column] < (q1 - 1.5 * iqr)) | (drop_df[numeric_column] > (q3 + 1.5 * iqr))
        c = c.any(axis=1)
        drop_df = drop_df.drop(drop_df[c].index, axis=0)
        drop_df.reset_index(inplace=True)
        drop_df.drop(['index'], axis=1, inplace=True)

        for column in categorical_column:
            if column not in ['work_phone', 'phone', 'email', 'car_reality']:
                encoder = LabelEncoder()
                encoder.fit(drop_df[column].values)
                drop_df[column] = pd.DataFrame(encoder.transform(drop_df[column].values).reshape(-1, 1), columns=[column])

        # oversampling
        feature = [True for _ in range(len(drop_df.columns) - 1)]
        feature[0] = False
        feature[1] = False
        feature[2] = False
        feature[3] = False
        sample = SMOTENC(categorical_features=feature, k_neighbors=3)
        df_X, df_Y = sample.fit_resample(drop_df.drop(['credit'], axis=1).values, drop_df[['credit']].values)
        df_Y = np.reshape(df_Y, (-1, 1))
        drop_df = pd.DataFrame(np.concatenate([df_X, df_Y], axis=1), columns=numeric_column + categorical_column)
        drop_df[numeric_column] = self.__z_score_nomalizer('train', drop_df[numeric_column])

        X = drop_df[numeric_column].values

        for column in categorical_column:
            one_hot_encoder = OneHotEncoder()
            one_hot_encoder.fit(drop_df[column].values.reshape((-1, 1)))
            label_train = one_hot_encoder.transform(drop_df[column].values.reshape(-1, 1)).toarray()
            X = np.concatenate([X, label_train], axis=1)

        y = X[:, -3:X.shape[1]]
        X = X[:, :-3]
        print('train preprocessing complete - oversampling')
        return X, y

    def train_data_oversampling_split(self, df:pd.DataFrame):
        drop_df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'credit']
        drop_df = drop_df[numeric_column + categorical_column]

        df_train, df_test = train_test_split(drop_df, test_size=0.33, random_state=4343)
        print(df_train.shape, df_test.shape)

        # 이상치 제거
        q1 = df_train[numeric_column].quantile(0.25)
        q3 = df_train[numeric_column].quantile(0.75)
        iqr = q3 - q1
        c = (df_train[numeric_column] < (q1 - 1.5 * iqr)) | (df_train[numeric_column] > (q3 + 1.5 * iqr))
        c = c.any(axis=1)
        df_train = df_train.drop(df_train[c].index, axis=0)
        df_train.reset_index(inplace=True)
        df_train.drop(['index'], axis=1, inplace=True)
        df_test.reset_index(inplace=True)
        df_test.drop(['index'], axis=1, inplace=True)

        for column in categorical_column:
            if column not in ['work_phone', 'phone', 'email', 'car_reality']:
                encoder = LabelEncoder()
                encoder.fit(self.__label_all_data[column])
                df_train[column] = pd.DataFrame(encoder.transform(df_train[column].values).reshape(-1, 1), columns=[column])
                df_test[column] = pd.DataFrame(encoder.transform(df_test[column].values).reshape(-1, 1), columns=[column])

        # oversampling
        feature = [True for _ in range(len(df_train.columns) - 1)]
        feature[0] = False
        feature[1] = False
        feature[2] = False
        feature[3] = False
        sample = SMOTENC(categorical_features=feature, k_neighbors=3)
        df_X, df_Y = sample.fit_resample(df_train.drop(['credit'], axis=1).values, df_train[['credit']].values)
        df_Y = np.reshape(df_Y, (-1, 1))
        df_train = pd.DataFrame(np.concatenate([df_X, df_Y], axis=1), columns=numeric_column + categorical_column)
        df_train[numeric_column] = self.__z_score_nomalizer('train', df_train[numeric_column])
        df_test[numeric_column] = self.__z_score_nomalizer('test', df_test[numeric_column])

        X_train = df_train[numeric_column].values
        X_test = df_test[numeric_column].values

        for column in categorical_column:
            all_d = []
            for i in [df_train[column].values, df_test[column].values]:
                c_d = []
                for d in i:
                    one_hot = [0 for _ in range(len(self.__label_all_data[column]))]
                    one_hot[int(d)] = 1
                    c_d.append(one_hot)
                all_d.append(c_d)
            label_train = np.asarray(all_d[0])
            X_train = np.concatenate([X_train, label_train], axis=1)
            label_test = np.asarray(all_d[1])
            X_test = np.concatenate([X_test, label_test], axis=1)

            # one_hot_encoder = OneHotEncoder()
            # one_hot_encoder.fit(np.asarray(range(len(self.__label_all_data[column]))).reshape((-1, 1)))
            # label_train = one_hot_encoder.transform(df_train[column].values.reshape(-1, 1)).toarray()
            # X_train = np.concatenate([X_train, label_train], axis=1)
            # label_test = one_hot_encoder.transform(df_test[column].values.reshape(-1, 1)).toarray()
            # X_test = np.concatenate([X_test, label_test], axis=1)

        Y_train = X_train[:, -3:X_train.shape[1]]
        X_train = X_train[:, :-3]
        Y_test = X_test[:, -3:X_test.shape[1]]
        X_test = X_test[:, :-3]
        print('train preprocessing complete - oversampling')
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        return X_train, X_test, Y_train, Y_test

    def test_data_preprocess(self, df:pd.DataFrame):
        drop_df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality']
        drop_df[numeric_column] = self.__z_score_nomalizer('test', drop_df[numeric_column])

        X = drop_df[numeric_column].values
        print(X.shape)
        for column in categorical_column:
            if column not in ['work_phone', 'phone', 'email', 'car_reality']:
                encoder = LabelEncoder()
                encoder.fit(drop_df[column].values)
                drop_df[column] = pd.DataFrame(encoder.transform(drop_df[column].values).reshape(-1, 1), columns=[column])
            one_hot_encoder = OneHotEncoder()
            one_hot_encoder.fit(drop_df[column].values.reshape(-1, 1))
            label = one_hot_encoder.transform(drop_df[column].values.reshape(-1, 1)).toarray()
            X = np.concatenate([X, label], axis=1)

        print('test preprocessing complete')
        return X