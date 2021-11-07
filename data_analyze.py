import DataIO

data_rw = DataIO.DataReadWrite()

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
data_train = data_rw.read_csv_to_df('train.csv')

# print(data_train.columns.size)

data_train = data_train.fillna('none')
print(data_train.isnull().mean())
data_train.drop(['index'], inplace=True, axis=1)
data_rw.save_df_to_csv(data_train, 'modified_train.csv')

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