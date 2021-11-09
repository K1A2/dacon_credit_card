import DataIO
import Preprocessing
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

data_rw = DataIO.DataReadWrite()
preprocessing = Preprocessing.Preprocess()

df = data_rw.read_csv_to_df('modified_train.csv')
df.drop(['index', 'FLAG_MOBIL'], axis=1, inplace=True)
numeric_column = ['Annual_income', 'working_day', 'begin_month']
categorical_column = ['gender', 'income_type', 'Education', 'family_type', 'house_type',
                      'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'credit']
# for c in categorical_column:
#     sns.countplot(x="credit", hue=c, data=df)
#     plt.title(c)
#     plt.show()
df['Education'] = LabelEncoder().fit_transform(df['Education'])
df['income_type'] = LabelEncoder().fit_transform(df['income_type'])

# a = []
# na = {}
# k = 0
# for i in ['Higher education' ,'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree']:
#     for j in ['Commercial associate', 'Working', 'State servant', 'Pensioner', 'Student']:
#         na[k] = i + ' ' + j
#         k+=1
# for i in range(df.shape[0]):
#     ed = df.loc[i, 'Education']
#     inc = df.loc[i, 'income_type']
#     a.append([na[ed * 5 + inc], df.loc[i, 'credit']])
# a = pd.DataFrame(a, columns=['income_edu', 'credit'])
# print(a.min())
# print(a.max())
# sns.countplot(x='credit', hue='income_edu', data=a, palette=sns.color_palette('deep', 25))
# plt.show()


# pd.options.display.max_rows = 60
# pd.options.display.max_columns = 20
print(df.columns.tolist())
for i in df.loc[df.duplicated(df.columns[:-1]),:].sort_values(df.columns[:-1].tolist()).values:
    print(list(i))
df.loc[df.duplicated(df.columns[:-1]),:].sort_values(df.columns[:-1].tolist()).to_csv('save.csv')




# df = df[numeric_column + ['DAYS_BIRTH'] + categorical_column]
#
# birth = np.abs(df['DAYS_BIRTH'].values)
# df.loc[:,'AGE'] = birth // 365
# df.drop(['DAYS_BIRTH'], inplace=True, axis=1)
#
# count = 0
# age_one_hot = []
# for i in [0, 20, 30, 40, 50, 60]:
#     df.loc[df['AGE'] >= i, 'AGE_RANGE'] = count
#     count += 1
# for i in df['AGE_RANGE']:
#     one = [0 for _ in range(6)]
#     one[int(i)] = 1
#     age_one_hot.append(one)
# age_one_hot = np.asarray(age_one_hot).reshape((-1, 6))
# print(df)
# df.drop(['AGE', 'AGE_RANGE'], axis=1, inplace=True)
#
# data = df[numeric_column].values
# data = np.concatenate([data, age_one_hot], axis=1)
#
# for column in categorical_column[:-1]:
#     if column not in ['work_phone', 'phone', 'email', 'car_reality']:
#         encoder = LabelEncoder()
#         d = encoder.fit_transform(df[column].values)
#         label = np.reshape(d, (-1, 1))
#     else:
#         label = df[column].values.reshape((-1, 1))
#     encoder = OneHotEncoder()
#     d = encoder.fit_transform(label)
#     data = np.concatenate([data, d.toarray()], axis=1)
#
# X = data
# y = df['credit'].values
#
# print(X.shape, y.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state = 434343)
#
# model = ExtraTreesClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict_proba(X_test)
# print(f"ExtraTreesClassifier log_loss: {log_loss(to_categorical(y_test), y_pred)}")
#
# model2 = HistGradientBoostingClassifier()
# model2.fit(X_train, y_train)
# y_pred = model2.predict_proba(X_test)
# loss = log_loss(to_categorical(y_test), y_pred)
# print(f"HistGradientBoostingClassifier log_loss: {loss}")
#
# model_cat = CatBoostClassifier()
# model_cat.fit(X_train, y_train)
# y_pred = model_cat.predict_proba(X_test)
# loss = log_loss(to_categorical(y_test), y_pred)
# print(f"CatBoostClassifier log_loss: {loss}")
#
#
# exit(0)
#
# df = data_rw.read_csv_to_df('test.csv')
# df.drop(['index', 'FLAG_MOBIL'], axis=1, inplace=True)
# df = df[numeric_column + categorical_column[:-1]]
# data = df[numeric_column].values
#
# for column in categorical_column[:-1]:
#     if column not in ['work_phone', 'phone', 'email', 'car_reality']:
#         encoder = LabelEncoder()
#         d = encoder.fit_transform(df[column].values)
#         label = np.reshape(d, (-1, 1))
#     else:
#         label = df[column].values.reshape((-1, 1))
#     encoder = OneHotEncoder()
#     d = encoder.fit_transform(label)
#     data = np.concatenate([data, d.toarray()], axis=1)
#
# X = data
#
# print(X.shape)
#
# pred_sub = model2.predict_proba(data)
# submission = pd.read_csv('./data/sample_submission.csv')
# submission.loc[:,1:] = pred_sub
# submission.to_csv(f'./data/submission/{loss}_xgboost.csv',index=False)

# print(model.feature_importances_)
# sns.barplot(y=model.feature_importances_, x=df.columns.tolist()[:-1])
# plt.show()

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
# data_train = data_rw.read_csv_to_df('train.csv')

# print(data_train.columns.size)

# data_train = data_train.fillna('none')
# print(data_train.isnull().mean())
# data_train.drop(['index'], inplace=True, axis=1)
# data_rw.save_df_to_csv(data_train, 'modified_train.csv')

# for column in data_train.columns:
#     if column in ['index', 'Annual_income', 'DAYS_BIRTH', 'working_day', 'begin_month']:
#         continue
#     print(data_train[column].value_counts())
#     print()
    # s = set()
    # for d in data_train[column]:
    #     s.add(d)
    # print(column, len(s), s)

# print(data_train.value_counts())