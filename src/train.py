import argparse

import mlens
import pandas as pd
from sklearn.metrics import r2_score

from model import HouseModel, DataModel
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

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=False, default='../data/train.csv',
                        help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", default="../models/model.pkl",
                        help="Куда сохранить обученную ML модель")
    parser.add_argument("--preprocess", default=True, type=lambda x: parse_bool(x),
                        help="Нужна ли предобработка данных")
    parser.add_argument("--split_data", default=True, type=lambda x: parse_bool(x),
                        help="Необходимо ли делить данные на обучающую/тестовую выборку")

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())
    train_df = pd.read_csv(args['d'])

    if args['preprocess']:
        preprocess_method = default_preprocess
    else:
        preprocess_method = None

    data_model = DataModel(train_df, preprocess_method)
    model = HouseModel()

    model.fit(data_model)
    predictions = model.predict(data_model.e_X_test)
    pickle_save(name=f'{args["mp"]}', obj=model)
    print(deviation_metric(data_model.e_y_test.values, predictions))
