import pickle
import Stacking
import category_encoders as ce

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier,Pool
import ParamTuning
pd.set_option('display.max_columns', None)

#Data

train_csv = pd.read_csv('./data/train.csv')
test_csv = pd.read_csv('./data/test.csv')
sample_submission_csv = pd.read_csv('./data/sample_submission.csv')

train_data = train_csv.copy()
test_data = test_csv.copy()

print(train_data.shape)
train_data.head()

numermic_col = []
cate_col = []

numermic_col = ['Annual_income','working_day','begin_month','DAYS_BIRTH']
default_numermic_col = ['Annual_income','working_day','begin_month','DAYS_BIRTH']
default_cate_col = ['gender','income_type','Education','family_type','house_type','work_phone','phone','email','occyp_type','car_reality']
cate_col = ['gender','income_type','Education','family_type','house_type','work_phone','phone','email','occyp_type','car_reality']
od_col = []

print(train_data.head())

train_data.fillna("None",inplace=True)
test_data.fillna("None",inplace=True)

train_data.drop(['index','FLAG_MOBIL'],inplace=True,axis=1)
test_data.drop(['index','FLAG_MOBIL'],inplace=True,axis=1)

train_data.loc[((train_data['working_day']== 0) & (train_data['occyp_type'] == "None")),'occyp_type'] = 'No_Job'
test_data.loc[((test_data['working_day']== 0) & (test_data['occyp_type'] == "None")),'occyp_type'] = 'No_Job'

train_data['DAYS_BIRTH'] = np.abs(train_data['DAYS_BIRTH'])
test_data['DAYS_BIRTH'] = np.abs(test_data['DAYS_BIRTH'])

train_data['working_day'] = np.abs(train_data['working_day'])
test_data['working_day'] = np.abs(test_data['working_day'])

train_data['begin_month'] = np.abs(train_data['begin_month'])
test_data['begin_month'] = np.abs(test_data['begin_month'])

########################################
# train_data
########################################
df = train_data

df['not_working_day'] = df['DAYS_BIRTH'] - df['working_day']
numermic_col.append('not_working_day')
df['age_y'] = df['DAYS_BIRTH'] // 365
numermic_col.append('age_y')
df['age_m'] = df['DAYS_BIRTH'] % 365 // 30
numermic_col.append('age_m')
df['age_w'] = df['DAYS_BIRTH'] % 365 % 30 // 7
numermic_col.append('age_w')

df['working_y'] = df['working_day'] // 365
numermic_col.append('working_y')
df['working_m'] = df['working_day'] % 365 // 30
numermic_col.append('working_m')
df['working_w'] = df['working_day'] % 365 % 30 // 7
numermic_col.append('working_w')

df['begin_y'] = df['begin_month'] // 12
numermic_col.append('begin_y')
df['begin_m'] = df['begin_month'] % 12
numermic_col.append('begin_m')
df['begin_prop_income'] = np.floor(df['Annual_income']  / df['begin_month'])
numermic_col.append('begin_prop_income')

df.replace(-np.inf,0,inplace=True)
df.replace(np.inf,0,inplace=True)
df.fillna(0,inplace=True)
train_data = df

########################################
# test_Data
########################################
df = test_data
df['not_working_day'] = df['DAYS_BIRTH'] - df['working_day']
df['age_y'] = df['DAYS_BIRTH'] // 365
df['age_m'] = df['DAYS_BIRTH'] % 365 // 30
df['age_w'] = df['DAYS_BIRTH'] % 365 % 30 // 7

df['working_y'] = df['working_day'] // 365
df['working_m'] = df['working_day'] % 365 // 30
df['working_w'] = df['working_day'] % 365 % 30 // 7

df['begin_y'] = df['begin_month'] // 12
df['begin_m'] = df['begin_month'] % 12
df['begin_prop_income'] = np.floor(df['Annual_income']  / df['begin_month'])

df.replace(-np.inf,0,inplace=True)
df.replace(np.inf,0,inplace=True)
df.fillna(0,inplace=True)
test_data = df

train_data_y = train_data['credit'].astype(int)
train_data.drop('credit',axis=1,inplace=True)
train_data.reset_index(inplace=True)
train_data.drop('index',axis=1,inplace=True)

for col in default_cate_col:
    minmax = LabelEncoder()
    train_data[col] = minmax.fit_transform(train_data[col])
    test_data[col] = minmax.transform(test_data[col])


income_range = [0, 180000, 330000, 490000, 640000, 800000, 950000, 1110000, 1260000, 1420000]
cnt = 0
for i in income_range:
    train_data.loc[train_data['Annual_income'] >= i, 'Annual_income_r'] = cnt
    test_data.loc[test_data['Annual_income'] >= i, 'Annual_income_r'] = cnt
    cnt += 1
birth_range = [0, 20 * 365, 30 * 365, 40 * 365, 50 * 365, 60 * 365]
cnt = 0
for i in birth_range:
    train_data.loc[train_data['DAYS_BIRTH'] >= i, 'DAYS_BIRTH_r'] = cnt
    test_data.loc[test_data['DAYS_BIRTH'] >= i, 'DAYS_BIRTH_r'] = cnt
    cnt += 1
work_range = [0, 1 * 365, 3 * 365, 5 * 365, 7 * 365, 10 * 365]
cnt = 0
for i in work_range:
    train_data.loc[train_data['working_day'] >= i, 'working_day_r'] = cnt
    test_data.loc[test_data['working_day'] >= i, 'working_day_r'] = cnt
    cnt += 1

from itertools import product

train_data['Owner'] = ''
test_data['Owner'] = ''
all_list = []
for col in ['gender','Annual_income','income_type','Education','family_type','house_type','DAYS_BIRTH','working_day','work_phone','phone','email','occyp_type','car_reality']:
    train_data['Owner'] = train_data['Owner']+ train_data[col].astype(int).astype(str)
    test_data['Owner'] = test_data['Owner'] + test_data[col].astype(int).astype(str)
for col in ['gender','Annual_income_r','income_type','Education','family_type','house_type','DAYS_BIRTH_r','working_day_r','work_phone','phone','email','occyp_type','car_reality']:
    all_list.append([i for i in range(len(train_data[col].unique().tolist()))])
train_data['Owner'] = train_data['Owner'].astype(str)
test_data['Owner'] = test_data['Owner'].astype(str)
cate_col.append('Owner')
all_list = list(product(*all_list))
print(all_list)
for i in range(len(all_list)):
    all_list[i] = ''.join([str(j) for j in all_list[i]])
print(all_list[0])
with open('./data/all_label', 'wb') as f:
    pickle.dump(all_list, f)
exit(0)
# with open('./data/params/best_param_lgbm_stacked', 'rb') as f:
#     params = pickle.load(f)
# print(params)

en = LabelEncoder()
en.fit(all_list)
train_data['Owner'] = en.transform(train_data['Owner'])
test_data['Owner'] = en.transform(test_data['Owner'])

print(train_data.head(10))


for col in numermic_col:
    minmax = StandardScaler()
    train_data[col] = minmax.fit_transform(train_data[[col]])
    test_data[col] = minmax.transform(test_data[[col]])
print(train_data)

print(cate_col)
print(train_data.columns.tolist())
print(test_data.columns.tolist())



# stacking = Stacking.StackingKfold(train_data, train_data_y, test_data, cate_col)
# stacking.stakcing(15)
# stacking.stacking_last(15)



# tunner = ParamTuning.Tuner(train_data, train_data_y, cate_col)
# param = tunner.tuning_catboost(50)
# print(param)
# with open('./data/params/last_catboost_param', 'wb') as f:
#     pickle.dump(param, f)

loss_list = []
pred = []
for i in [15]:
    n_split = i
    sk_fold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=4558)
    t_model = 0
    cnt = 0
    loss_sum = []
    fin_pred = []
    for train_idx, test_idx in sk_fold.split(X=train_data, y=train_data_y):
        cnt += 1
        x_train, x_val = train_data.loc[train_idx], train_data.loc[test_idx]
        y_train, y_val = train_data_y[train_idx], train_data_y[test_idx]
        x_test = test_data.copy()

        model = CatBoostClassifier(n_estimators=10000)
        train_pool = Pool(x_train, y_train, cat_features=cate_col)
        val_pool = Pool(x_val, y_val, cat_features=cate_col)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=50, use_best_model=True)
        t_model = model
        pred_test = model.predict_proba(x_val)
        pred.append(model.predict_proba(x_test))
        score = log_loss(y_val, pred_test)
        print(f"{cnt} : logloss {score}")
        loss_sum.append(score)

    print(f"mean log loss {sum(loss_sum) / n_split}")
    loss_list.append(sum(loss_sum) / n_split)

print(loss_list)

fin_pred = np.asarray(pred).sum(axis=0)
fin_pred /= 15
output = pd.concat((sample_submission_csv['index'], pd.DataFrame(fin_pred, columns=[0, 1, 2])), axis=1)
output.to_csv(f'./data/{loss_list[0]}_catboost.csv', index=False)