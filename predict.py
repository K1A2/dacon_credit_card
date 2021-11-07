import pandas as pd
import tensorflow.keras as keras
import DataIO
import Preprocessing
import numpy as np
import os

test_data_name = 'test.csv'
mode = 1

def main():
    model = keras.models.load_model('data/models/model3_outlier_adam2/16_0_7560163.h5')
    datarw = DataIO.DataReadWrite()
    prepeocess = Preprocessing.Preprocess()
    test_data = datarw.read_csv_to_df(test_data_name)
    X = prepeocess.test_data_preprocess(test_data)

    input_data_size = []
    if mode == 0:  # multi_input_1
        input_data_size = [4, 2, 5, 5, 5, 6, 2, 2, 2, 19, 3]
    elif mode == 1 or mode == 2:  # multi_input_2
        input_data_size = [4, 51]

    index = 0
    X_train_input = []
    for i in input_data_size:
        X_train_input.append(X[:, index: index + i])
        index += i

    result = model.predict(X_train_input).tolist()
    index = test_data['index'].values.tolist()
    for i in range(len(index)):
        result[i] = [index[i]] + result[i]

    # for i in model.predict(X_train_input):
    #     class_num = np.argmax(i)
    #     one_hot = [0 for _ in range(3)]
    #     one_hot[class_num] = 1
    #     result.append(one_hot)

    result = pd.DataFrame(result, columns=['index', '0', '1', '2'])
    print(result)
    datarw.save_df_to_csv(result, f'submission/submission_{len(os.listdir("./data/submission/"))}.csv')

if __name__ == '__main__':
    main()