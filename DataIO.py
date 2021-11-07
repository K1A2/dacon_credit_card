import pandas as pd

class DataReadWrite:
    __default_path = './data/'

    def read_csv_to_df(self, file_name):
        data = pd.read_csv(self.__default_path + file_name)
        return data

    def save_df_to_csv(self, df:pd.DataFrame, file_name):
        df.to_csv(self.__default_path + file_name, index=False)