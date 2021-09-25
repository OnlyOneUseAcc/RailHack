import argparse

import mlens
import pandas as pd
from sklearn.metrics import r2_score

from model import HouseModel
from settings import NUM_FEATURES
from metrics import deviation_metric
from sklearn.model_selection import train_test_split
import pickle
from mlens.utils.utils import pickle_save
from utils import default_preprocess


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
    y = train_df['per_square_meter_price']
    if args['preprocess']:
        X = default_preprocess(X)
        print('pre proc finish')
        pass

    model = HouseModel()

    if not args['split_data']:
        model.fit(X, y)
        print('finish fit')
        pickle_save(name=f'{args["mp"]}', obj=model)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        pickle_save(name=f'{args["mp"]}', obj=model)
        print(deviation_metric(y_test.values, predictions))
