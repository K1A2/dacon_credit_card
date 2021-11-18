from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
import category_encoders as ce

'''
gender random_under {'F', 'M'}
income_type 5 {'Commercial associate', 'Student', 'State servant', 'Pensioner', 'Working'}
Education 5 {'Lower secondary', 'Academic degree', 'Higher education', 'Incomplete higher', 'Secondary / secondary special'}
family_type 5 {'Married', 'Separated', 'Single / not married', 'Widow', 'Civil marriage'}
house_type 6 {'Office apartment', 'With parents', 'House / apartment', 'Rented apartment', 'Municipal apartment', 'Co-op apartment'}
FLAG_MOBIL none {none}
work_phone random_under {0, none}
phone random_under {0, none}
email random_under {0, none}
occyp_type 19 {'Security staff', 'Laborers', 'HR staff', 'Cleaning staff', 'Waiters/barmen staff', 'Medicine staff', 'Managers', 'Accountants', 'IT staff', 'Realty agents', 'Low-skill Laborers', 'Private service staff', 'none', 'Core staff', 'Drivers', 'Secretaries', 'Sales staff', 'Cooking staff', 'High skill tech staff'}
car_reality under_tome {0, none, random_under}
credit under_tome {0.0, none.0, random_under.0}
'''

default_path = './data/'
label_all_data = {'gender': ['F', 'M'],
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
                    'Age': [0, 20, 30, 40, 50, 60],
                    'work_phone': [0, 1], 'phone': [0, 1], 'email': [0, 1], 'car_reality': [0, 1, 2],
                    'credit': [0.0, 1.0, 2.0]}


def gauss_nomalizer(type, df: pd.DataFrame):
    scaler = GaussRankScaler()
    if type == 'test':
        with open(default_path + 'gauss_score', 'rb') as f:
            scaler = pickle.load(f)
    else:
        with open(default_path + 'gauss_score', 'wb') as f:
            scaler.fit(df.values)
            pickle.dump(scaler, f)
    return pd.DataFrame(scaler.transform(df.values), columns=df.columns.tolist())

def z_score_nomalizer(type, df: pd.DataFrame):
    data = [df.mean(), df.std()]
    if type == 'test':
        with open(default_path + 'z_score', 'rb') as f:
            data = pickle.load(f)
    else:
        with open(default_path + 'z_score', 'wb') as f:
            pickle.dump(data, f)
    return (df - data[0]) / data[1]


class Preprocess():

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
        print('make plot finish')

from GaussRankScaler import GaussRankScaler
class PreprocesserGBoost():
    def data_preprocess_2(self, df:pd.DataFrame, type):
        # 쓸데없는 데이터 제거
        df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality']
        df = df.fillna('none')

        X = np.zeros((df.shape[0], 1))
        # LabelEncoding, OneHotEncoding
        for column in categorical_column:
            encoder = label_all_data[column]
            if column not in  ['work_phone', 'phone', 'email', 'car_reality']:
                encoding = []
                for d in df[column].values:
                    encoding.append(encoder.index(d))
                df.loc[:,column] = np.asarray(encoding)
            encoding = []
            for d in df[column].values:
                e = [0 for _ in range(len(encoder))]
                e[d] = 1
                encoding.append(e)
            X = np.concatenate([X, np.asarray(encoding)], axis=1)
        X = X[:, 1:]


        df.loc[:,'not_working_day'] = np.abs(df['DAYS_BIRTH'].values) - np.abs(df['working_day'].values)
        not_working_day = df['not_working_day'].values
        df.loc[:, 'not_working_year'] = not_working_day // 365
        not_working_day %= 365
        df.loc[:, 'not_working_month'] = not_working_day // 30
        not_working_day %= 30
        df.loc[:, 'not_working_week'] = not_working_day // 7

        # year_threshold = [0, 2, 5, 8, 11, 15, 20]
        # for i in range(len(year_threshold)):
        #     df.loc[df['not_working_year'] >= year_threshold[i], 'not_working_year_range'] = i
        # d = []
        # for i in df['not_working_year_range'].values:
        #     one = [0 for _ in range(len(year_threshold))]
        #     one[int(i)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)
        # d = []
        # for i in df['not_working_month'].values:
        #     one = [0 for _ in range(12)]
        #     one[int(i - 1)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)
        # d = []
        # for i in df['not_working_week'].values:
        #     one = [0 for _ in range(7)]
        #     one[int(i - 1)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)


        # birth -> age
        birth = np.abs(df['DAYS_BIRTH'].values)
        df.loc[:, 'Age'] = birth // 365

        # age_threshold = [0, 20, 30, 40, 50, 60]
        # for i in range(len(age_threshold)):
        #     df.loc[df['Age'] >= age_threshold[i], 'Age_range'] = i
        # d = []
        # for i in df['Age_range'].values:
        #     one = [0 for _ in range(len(age_threshold))]
        #     one[int(i)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)


        # working_day -> year, month, day
        working_day = np.abs(df['working_day'].values)
        df.loc[:, 'working_year'] = working_day // 365
        working_day %= 365
        df.loc[:, 'working_month'] = working_day // 30
        working_day %= 30
        df.loc[:, 'working_week'] = working_day // 7

        # year_threshold = [0, 2, 5, 8, 11, 15, 20]
        # for i in range(len(year_threshold)):
        #     df.loc[df['working_year'] >= year_threshold[i], 'working_year_range'] = i
        # d = []
        # for i in df['working_year_range'].values:
        #     one = [0 for _ in range(len(year_threshold))]
        #     one[int(i)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)
        # d = []
        # for i in df['working_month'].values:
        #     one = [0 for _ in range(12)]
        #     one[int(i - 1)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)
        # d = []
        # for i in df['working_week'].values:
        #     one = [0 for _ in range(7)]
        #     one[int(i - 1)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)


        # begin_month -> year, month:
        begin_month = np.abs(df['begin_month'].values)
        df.loc[:, 'begin_year'] = begin_month // 12
        begin_month %= 12
        df.loc[:, 'begin_month'] = begin_month

        # year_threshold = [0, 1, 2, 3, 4, 5]
        # for i in range(len(year_threshold)):
        #     df.loc[df['begin_year'] >= year_threshold[i], 'begin_year_range'] = i
        # d = []
        # for i in df['begin_year_range'].values:
        #     one = [0 for _ in range(len(year_threshold))]
        #     one[int(i)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)
        # d = []
        # for i in df['begin_month'].values:
        #     one = [0 for _ in range(12)]
        #     one[int(i - 1)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)


        # Annual_income -> Annual_income_range
        # incoume_threshold = list(range(0, 1000001, 15000))
        # for i in range(len(incoume_threshold)):
        #     df.loc[df['Annual_income'] >= incoume_threshold[i], 'Annual_income_range'] = i
        # d = []
        # for i in df['Annual_income_range'].values:
        #     one = [0 for _ in range(len(incoume_threshold))]
        #     one[int(i)] = 1
        #     d.append(one)
        # X = np.concatenate([X, np.asarray(d)], axis=1)

        numeric_column = ['Annual_income', 'begin_year', 'begin_month', 'working_year', 'working_month', 'working_week',
            'Age', 'not_working_year', 'not_working_month', 'not_working_week']
        # df[['Annual_income']] = gauss_nomalizer(type, df[['Annual_income']])
        # X = np.concatenate([X, df[['Annual_income']].values], axis=1)
        df[numeric_column] = z_score_nomalizer(type, df[numeric_column])
        X = np.concatenate([X, df[numeric_column].values], axis=1)

        print(X.shape)
        print(X)
        return X

    def data_preprocess_2_label(self, df:pd.DataFrame, typed):
        # 쓸데없는 데이터 제거
        df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality']
        df = df.fillna('none')

        X = np.zeros((df.shape[0], 1))
        # LabelEncoding, OneHotEncoding
        for column in categorical_column:
            encoder = label_all_data[column]
            if column not in  ['work_phone', 'phone', 'email', 'car_reality']:
                encoding = []
                for d in df[column].values:
                    encoding.append(encoder.index(d))
                df.loc[:,column] = np.asarray(encoding)
            X = np.concatenate([X, df.loc[:,column].values.reshape(-1,1)], axis=1)
        X = X[:, 1:]

        print(X.shape)
        print(X)
        return X

    def data_preprocess_2_comb(self, df:pd.DataFrame, typed):
        from itertools import product
        relation_income_edu_occu = list(product(*[label_all_data['income_type'], label_all_data['Education'],  label_all_data['occyp_type']]))
        relation_gender_occu = list(product(*[label_all_data['gender'], label_all_data['occyp_type']]))
        relation_family_house = list(product(*[ label_all_data['family_type'], label_all_data['house_type']]))
        relation_all = list(product(*[label_all_data['income_type'], label_all_data['Education'],
                                      label_all_data['family_type'], label_all_data['house_type'], label_all_data['occyp_type']]))
        # relation_owner = list(product(*[label_all_data['gender'], label_all_data['income_type'], label_all_data['Education'],
        #                                 label_all_data['family_type'], label_all_data['house_type'], label_all_data['occyp_type'],
        #                                 label_all_data['DAYS_BIRTH']]))
        relation_phone_mail = list(product(*[label_all_data['phone'], label_all_data['work_phone'], label_all_data['email']]))
        relation_occupy_car = list(product(*[label_all_data['occyp_type'], label_all_data['car_reality']]))
        relation_gender_car = list(product(*[label_all_data['gender'], label_all_data['car_reality']]))
        label_all_data['family_type_same'] = [0,1]
        label_all_data['money_relation'] = [' '.join([str(j) for j in i]) for i in relation_income_edu_occu]
        label_all_data['gneder_occuyp'] = [' '.join([str(j) for j in i]) for i in relation_gender_occu]
        label_all_data['family_house'] = [' '.join([str(j) for j in i]) for i in relation_family_house]
        label_all_data['relation_giefho'] = [' '.join([str(j) for j in i]) for i in relation_all]
        # label_all_data['owner'] = [' '.join([str(j) for j in i]) for i in relation_owner]
        label_all_data['phone_mail'] = [' '.join([str(j) for j in i]) for i in relation_phone_mail]
        label_all_data['occuyp_car'] = [' '.join([str(j) for j in i]) for i in relation_occupy_car]
        label_all_data['gender_reality'] = [' '.join([str(j) for j in i]) for i in relation_gender_car]

        # 쓸데없는 데이터 제거
        df = df.drop(['index', 'FLAG_MOBIL'], axis=1)
        numeric_column = ['Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                              'work_phone', 'phone', 'email', 'occyp_type', 'car_reality']

        df = df.fillna('none')

        df.loc[:, 'money_relation'] = df['income_type'] + ' ' + df['Education'] + ' ' + df['occyp_type']
        # df.loc[:, 'gneder_occuyp'] = df['gender'] + ' ' + df['occyp_type']
        df.loc[:, 'family_house'] = df['family_type'] + ' ' + df['house_type']
        df.loc[:, 'relation_giefho'] = df['income_type'] + ' ' + df['Education'] + ' ' +\
                                    df['family_type'] + ' ' + df['house_type'] + ' ' + df['occyp_type']
        # df.loc[:, 'owner'] = df['gender'] + ' ' + df['income_type'] + ' ' + df['Education'] + ' ' +\
        #                     df['family_type'] + ' ' + df['house_type'] + ' ' + df['occyp_type']
        df.loc[:, 'phone_mail'] = df['phone'].astype(str) + ' ' + df['work_phone'].astype(str) + ' ' + df['email'].astype(str)
        df.loc[:, 'occuyp_car'] = df['occyp_type'] + ' ' + df['car_reality'].astype(str)
        df.loc[:, 'gender_reality'] = df['gender'].astype(str) + " " + df['car_reality'].astype(str)
        df.loc[:,'family_type_same'] = 0
        df.loc[(df['family_type'] == 'Married') | (df['family_type'] == 'Civil marriage'), 'family_type_same'] = 1

        # for relation in relation_income_edu_occu:
        #     df.loc[(df['income_type'] == relation[0]) &
        #            (df['Education'] == relation[1]) &
        #            (df['occyp_type'] == relation[2]),'money_relation'] = f'{relation[0]}_{relation[1]}_{relation[2]}'
        # for relation in relation_gender_occu:
        #     df.loc[(df['gender'] == relation[0]) &
        #            (df['occyp_type'] == relation[1]), 'gneder_occuyp'] = f'{relation[0]}_{relation[1]}'
        # for relation in relation_family_house:
        #     df.loc[(df['gender'] == relation[0]) &
        #            (df['family_type'] == relation[1]) &
        #            (df['house_type'] == relation[2]), 'family_house'] = f'{relation[0]}_{relation[1]}_{relation[2]}'
        # for relation in relation_all:
        #     df.loc[(df['gender'] == relation[0]) &
        #            (df['income_type'] == relation[1]) &
        #            (df['Education'] == relation[2]) &
        #            (df['family_type'] == relation[3]) &
        #            (df['house_type'] == relation[4]) &
        #            (df['occyp_type'] == relation[5]), 'relation_all'] = \
        #             f'{relation[0]}_{relation[1]}_{relation[2]}_{relation[3]}_{relation[4]}_{relation[5]}'

        # df = df.drop(['income_type', 'Education', 'occyp_type', 'gender', 'family_type',
        #               'house_type', 'work_phone', 'phone', 'email', 'car_reality'], axis=1)

        df['DAYS_BIRTH'] = np.abs(df['DAYS_BIRTH'].values)
        df['working_day'] = np.abs(df['working_day'].values)
        df['begin_month'] = np.abs(df['begin_month'].values)
        df['not_working_day'] = df['DAYS_BIRTH'] - df['working_day']

        df['age_y'] = df['DAYS_BIRTH'] // 365
        df['age_m'] = df['DAYS_BIRTH'] % 365 // 30
        df['age_w'] = df['DAYS_BIRTH'] % 365 % 30 // 7
        df['working_y'] = df['working_day'] // 365
        df['working_m'] = df['working_day'] % 365 // 30
        df['working_w'] = df['working_day'] % 365 % 30 // 7
        df['not_working_y'] = df['not_working_day'] // 365
        df['not_working_m'] = df['not_working_day'] % 365 // 30
        df['not_working_w'] = df['not_working_day'] % 365 % 30 // 7
        df['begin_y'] = df['begin_month'] // 12
        df['begin_m'] = df['begin_month'] % 12
        df['all_income'] = df['Annual_income'] * df['working_y'] + df['Annual_income'] / 12 * df['working_m'] +\
                           df['Annual_income'] / 12 / 4 * df['working_w']
        df['make_card_working'] = df['working_day'] // 30 - df['begin_month']
        df['make_card_working_y'] = df['make_card_working'] // 12
        df['make_card_working_m'] = df['make_card_working'] % 12
        df['make_card_birth'] = df['DAYS_BIRTH'] // 30 - df['begin_month']
        df['make_card_birth_y'] = df['make_card_birth'] // 12
        df['make_card_birth_m'] = df['make_card_birth'] % 12

        numeric_column = ['age_y', 'age_m', 'age_w', 'working_y', 'working_m', 'working_w', 'not_working_y',
                          'not_working_m', 'not_working_w', 'begin_y', 'begin_m', 'all_income',
                          'make_card_working', 'make_card_working_y', 'make_card_working_m',
                          'make_card_birth', 'make_card_birth_y', 'make_card_birth_m', 'not_working_day',
                          'Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']
        categorical_column = ['money_relation', 'gneder_occuyp', 'family_house', 'relation_giefho', 'family_type_same',
                              'phone_mail', 'occuyp_car', 'gender', 'income_type', 'Education', 'family_type',
                              'house_type', 'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'gender_reality']

        categorical_column.remove('gneder_occuyp')

        if typed == 'train':
            categorical_column.append('credit')

        # income = [0, 50000, 100000, 150000, 200000, 250000, 300000, 400000, 600000, 800000]
        # m = 0
        # for i in income:
        #     df.loc[df['Annual_income'] >= i, 'income_cat'] = m
        #     m += 1
        # age = [0, 20, 30, 40, 50, 60]
        # m = 0
        # for i in age:
        #     df.loc[df['age_y'] >= i, 'age_cat'] = m
        #     m += 1
        # relation_age_income = list(product(*[np.arange(0.0, len(income)).tolist(), np.arange(0.0, len(age)).tolist()]))
        # label_all_data['age_income'] = [' '.join([str(j) for j in i]) for i in relation_age_income]
        # df.loc[:, 'age_income'] = df['income_cat'].astype(str) + ' ' + df['age_cat'].astype(str)
        # relation_age_income_ocu = list(product(*[np.arange(0.0, len(income)).tolist(), np.arange(0.0, len(age)).tolist(),
        #                                          label_all_data['occyp_type']]))
        # label_all_data['age_income_occu'] = [' '.join([str(j) for j in i]) for i in relation_age_income_ocu]
        # df.loc[:, 'age_income_occu'] = df['income_cat'].astype(str) + ' ' + df['age_cat'].astype(str) + ' ' + df['occyp_type']
        # relation_income_car = list(product(*[np.arange(0.0, len(income)).tolist(), np.arange(0.0, len(age)).tolist(),
        #                                      label_all_data['car_reality']]))
        # label_all_data['income_age_car'] = [' '.join([str(j) for j in i]) for i in relation_income_car]
        # df.loc[:, 'income_age_car'] = df['income_cat'].astype(str) + ' ' + df['age_cat'].astype(str) + ' ' + df['car_reality'].astype(str)
        # numeric_column += ['make_card_working', 'make_card_working_y', 'make_card_working_m',
        #                   'make_card_birth', 'make_card_birth_y', 'make_card_birth_m']
        # categorical_column += ['age_income', 'age_income_occu', 'income_age_car']



        df = df.drop(['gender', 'work_phone', 'phone', 'email', 'car_reality'], axis=1)
        for i in ['gender', 'work_phone', 'phone', 'email', 'car_reality']:
            categorical_column.remove(i)

        print(df.iloc[0,:])

        if typed == 'train':
            categorical_column.remove('credit')
            y = df[['credit']].astype(int)
            df = df.drop(['credit'], axis=1)

        for column in categorical_column:
            # e = ce.TargetEncoder()
            e = LabelEncoder()
            if typed == 'test':
                with open(default_path + f'encoder/label_encoding_{column}', 'rb') as f:
                    e = pickle.load(f)
            else:
                with open(default_path + f'encoder/label_encoding_{column}', 'wb') as f:
                    e.fit(label_all_data[column])
                    pickle.dump(e, f)
            df[column] = e.transform(df[column].values)

        # categorical_column += ['income_cat', 'age_cat']

            # cat = []
            # for c in df.columns:
            #     if c in categorical_column:
            #         cat.append(True)
            #     else:
            #         cat.append(False)
            # over_sampling = SMOTENC(categorical_features=cat, random_state=42)
            # df, y = over_sampling.fit_resample(df, y)

        for n in numeric_column:
            df[n] = np.floor(df[n].values).astype(int)

        df[numeric_column] = z_score_nomalizer(typed, df[numeric_column])
        # df = df.drop(['DAYS_BIRTH', 'working_day', 'begin_month', 'not_working_day', 'Annual_income'], axis=1)

        # if typed == 'train':
        #     df = df.drop(df[(df['all_income'] < df['all_income'].quantile(0.25) - 1.5 * (
        #                 df['all_income'].quantile(0.75) - df['all_income'].quantile(0.25)))].index, axis=0)
        #     print(df)

        # for c in numeric_column:
        #     sns.histplot(x=df[c])
        #     plt.show()

        for i in categorical_column:
            df[i] = df[i].astype(int)

        print(df.iloc[0,:])
        print(df.shape)
        if typed == 'train':
            return df, y, numeric_column, categorical_column
        else:
            return df, numeric_column, categorical_column