import DataIO
import Preprocessing
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss

def main():
    dataio = DataIO.DataReadWrite()
    preprocesser = Preprocessing.PreprocesserGBoost()
    df = dataio.read_csv_to_df('train.csv')
    df_test = dataio.read_csv_to_df('test.csv')

    # 모든 데이터가 중복 되는 열 제거
    df = df.drop_duplicates(df.columns)

    # 클래스 분리
    y = df.iloc[:,-1]
    df = df.drop(['credit'], axis=1)

    # 데이터 전처리
    X = preprocesser.data_preprocess_2(df)
    X_submmit = preprocesser.data_preprocess_2(df_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=2424)

    model = CatBoostClassifier(logging_level='Silent')
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred = model.predict_proba(X_test)
    print(f"CatBoostClassifier log_loss: {log_loss(to_categorical(y_test), y_pred)}")

if __name__ == '__main__':
    main()