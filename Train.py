import DataIO
import Model
import Preprocessing
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

train_data_name = 'modified_train.csv'
mode = 2
graph = False

def main():
    datarw = DataIO.DataReadWrite()
    prepeocess = Preprocessing.Preprocess()
    train_data = datarw.read_csv_to_df(train_data_name)

    # if only make graph
    if graph:
        prepeocess.make_analyze_plot(train_data)
        exit(0)

    # X, y = prepeocess.train_data_preprocess(train_data)
    X, y = prepeocess.train_data_oversampling(train_data)
    # X_train, X_test, y_train, y_test = prepeocess.train_data_oversampling(train_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4343)

    input_data_size = []
    if mode == 0: # multi_input_1
        input_data_size = [4, 2, 5, 5, 5, 6, 2, 2, 2, 19, 3]
    elif mode == 1 or mode == 2: # multi_input_2
        input_data_size = [4, 51]

    X_train_input = []
    X_test_input = []
    index = 0
    for i in input_data_size:
        X_train_input.append(X_train[:,index: index + i])
        X_test_input.append(X_test[:,index: index + i])
        index += i

    model_loader = Model.Models()

    if mode == 0: # multi_input_1
        model = model_loader.multi_input_model_builder_1()
    elif mode == 1: # multi_input_2
        model = model_loader.multi_input_model_builder_2()
    elif mode == 2:
        model = model_loader.multi_input_model_builder_3()

    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])
    print(model.summary())
    # plot_model(model, to_file='data/plot_model/model.png', show_shapes=True, show_layer_names=True)

    # class_weight
    # history = model.fit(X_train_input, y_train, epochs=100, batch_size=64, class_weight={0:0.4391087424878104, none:0.3815625354348566, random_under:0.17932872207733302},
    #                     validation_data=[X_test_input, y_test], callbacks=[Model.AutoSaveCallback()])

    history = model.fit(X_train_input, y_train, epochs=100, batch_size=64,
                        validation_data=[X_test_input, y_test], callbacks=[Model.AutoSaveCallback()])

if __name__ == '__main__':
    main()