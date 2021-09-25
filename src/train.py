import argparse
import pandas as pd
from sklearn.metrics import r2_score

from model import HouseModel
from sklearn.model_selection import train_test_split
import pickle



def parse_bool(str_argument):
    if str_argument.lower() == 'true' or str_argument == '1':
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Скрипт для обучения модели",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True,
                        help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", default="models/model.pkl",
                        help="Куда сохранить обученную ML модель")
    parser.add_argument("--preprocess", default=True, type=lambda x: parse_bool(x),
                        help="Нужна ли предобработка данных")
    parser.add_argument("--split_data", default=False, type=lambda x: parse_bool(x),
                        help="Необходимо ли делить данные на обучающую/тестовую выборку")

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())
    train_df = pd.read_csv(args['d'])
    X = train_df.drop(columns='per_square_meter_price')
    y = train_df['train_df']
    if args['preprocess']:

        pass

    model = HouseModel()

    if not args['split_data']:
        model.fit(X, y)
        with open(f'{args["mp"]}', 'wb') as model_file:
            pickle.dump(model, model_file)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        model.fit(X_train, y_train)

        with open(f'{args["mp"]}', 'wb') as model_file:
            pickle.dump(model, model_file)

        predictions = model.predict(X_test)

        print(r2_score(y_test, predictions))

    pass