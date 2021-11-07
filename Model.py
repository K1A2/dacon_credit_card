import tensorflow
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Input, concatenate, Reshape, Conv1D, Flatten, Dropout
import os

# ['gender', 'income_type', 'Education', 'family_type', 'house_type',
# 'work_phone', 'phone', 'email', 'occyp_type', 'car_reality', 'credit']
class Models():
    def multi_input_model_builder_1(self):
        inputs_numeric = Input(shape=(4,))
        numeric_dens = Dense(16, activation='relu')(inputs_numeric)
        numeric_dens = Dense(8, activation='relu')(numeric_dens)

        input_gender = Input(shape=(2,))
        gender_dens = Dense(8, activation='relu')(input_gender)

        input_income_type = Input(shape=(5,))
        income_type_dens = Dense(16, activation='relu')(input_income_type)
        income_type_dens = Dense(8, activation='relu')(income_type_dens)

        input_education = Input(shape=(5,))
        education_type_dens = Dense(16, activation='relu')(input_education)
        education_type_dens = Dense(8, activation='relu')(education_type_dens)

        input_family_type = Input(shape=(5,))
        family_type_dens = Dense(16, activation='relu')(input_family_type)
        family_type_dens = Dense(8, activation='relu')(family_type_dens)

        input_house_type = Input(shape=(6,))
        house_type_dens = Dense(16, activation='relu')(input_house_type)
        house_type_dens = Dense(8, activation='relu')(house_type_dens)

        input_work_phone = Input(shape=(2,))
        work_phone_dens = Dense(8, activation='relu')(input_work_phone)

        input_phone = Input(shape=(2,))
        phone_dens = Dense(8, activation='relu')(input_phone)

        input_email = Input(shape=(2,))
        email_dens = Dense(8, activation='relu')(input_email)

        input_occyp_type = Input(shape=(19,))
        occyp_type_dens = Dense(32, activation='relu')(input_occyp_type)
        occyp_type_dens = Dense(16, activation='relu')(occyp_type_dens)
        occyp_type_dens = Dense(8, activation='relu')(occyp_type_dens)

        input_car_reality = Input(shape=(3,))
        car_reality_dens = Dense(8, activation='relu')(input_car_reality)

        categorical_dens = concatenate([gender_dens, income_type_dens, education_type_dens, family_type_dens,
                                        house_type_dens, work_phone_dens, phone_dens, email_dens, occyp_type_dens,
                                        car_reality_dens])
        categorical_dens = Dense(128, activation='relu')(categorical_dens)
        categorical_dens = Dense(64, activation='relu')(categorical_dens)
        categorical_dens = Dense(32, activation='relu')(categorical_dens)
        categorical_dens = Dense(16, activation='relu')(categorical_dens)

        final_dens = concatenate([numeric_dens, categorical_dens])
        final_dens = Dense(32, activation='relu')(final_dens)
        final_dens = Dense(16, activation='relu')(final_dens)
        final_dens = Dense(3, activation='softmax')(final_dens)

        model = Model([inputs_numeric, input_gender, input_income_type, input_education, input_family_type,
                       input_house_type, input_work_phone, input_phone, input_email, input_occyp_type,
                       input_car_reality], final_dens)
        return model

    def multi_input_model_builder_2(self):
        input_numeric = Input(shape=(4,))
        numerica_dense = Dense(32)(input_numeric)

        input_categorical = Input(shape=(51,))
        categotrical_dense = Reshape((17, 3))(input_categorical)
        categotrical_dense = Conv1D(20, 5, activation='relu', input_shape=(17, 3))(categotrical_dense)
        categotrical_dense = Flatten()(categotrical_dense)
        categotrical_dense = Dense(512, activation='relu')(categotrical_dense)
        categotrical_dense = Dense(256, activation='relu')(categotrical_dense)
        categotrical_dense = Dense(128, activation='relu')(categotrical_dense)
        categotrical_dense = Dropout(0.2)(categotrical_dense)
        categotrical_dense = Dense(64, activation='relu')(categotrical_dense)

        final_dense = concatenate([numerica_dense, categotrical_dense])
        final_dense = Dense(128, activation='relu')(final_dense)
        final_dense = Dense(64, activation='relu')(final_dense)
        final_dense = Dropout(0.2)(final_dense)
        final_dense = Dense(32, activation='relu')(final_dense)
        final_dense = Dense(16, activation='relu')(final_dense)
        final_dense = Dense(3, activation='softmax')(final_dense)

        model = Model([input_numeric, input_categorical], final_dense)
        return model

    def multi_input_model_builder_3(self):
        input_numeric = Input(shape=(4,))
        numerica_dense = Dense(16)(input_numeric)

        input_categorical = Input(shape=(51,))
        categotrical_dense = Dense(128, activation='relu')(input_categorical)
        categotrical_dense = Dense(64, activation='relu')(categotrical_dense)

        final_dense = concatenate([numerica_dense, categotrical_dense])
        final_dense = Dense(128, activation='relu')(final_dense)
        final_dense = Dense(64, activation='relu')(final_dense)
        final_dense = Dense(32, activation='relu')(final_dense)
        final_dense = Dense(16, activation='relu')(final_dense)
        final_dense = Dense(3, activation='softmax')(final_dense)

        model = Model([input_numeric, input_categorical], final_dense)
        return model

class AutoSaveCallback(Callback):
    loss = 1
    __save_path = './data/models/'
    def __init__(self):
        self.__save_path += f'{len(os.listdir(self.__save_path)) + 1}/'
        os.makedirs(self.__save_path)

    def on_epoch_end(self, epoch, logs=None):
        now_loss = logs['val_categorical_crossentropy']
        if self.loss > now_loss:
            save_model(self.model, self.__save_path + f'{epoch}_{str(now_loss).replace(".", "_")}.h5')
            self.loss = now_loss