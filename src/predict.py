import argparse
import pandas as pd

from utils import test_preproc
from settings import NUM_FEATURES
from mlens.utils.utils import pickle_load
from metrics import deviation_metric
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description="Скрипт для предсказания модели",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--test_data", "-d", type=str, dest="d", required=False, default='../data/test.csv',
                        help="Путь до отложенной выборки")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", default="../models/model.pkl",
                        help="Пусть до сериализованной ML модели")
    parser.add_argument("--output", "-o", type=str, dest="o", default='../data/test_submission.csv',
                        help="Путь до выходного файла")

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())
    # test_ones_data = pd.read_csv('../data/test.csv')
    # test_ones_data = test_ones_data[test_ones_data['price_type'] == 1]
    # target_ones = test_ones_data[['id', 'per_square_meter_price']]
    # test_ones_data = test_ones_data.drop(columns='per_square_meter_price')
    test_data = pd.read_csv(args['d'])
    id_data = test_data['id']

    with open('../models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    test_ones_data = test_preproc(test_data, scaler=scaler)
    model = pickle_load(name=f'{args["mp"]}')
    test_ones_data['per_square_meter_price'] = model.predict(test_ones_data)
    # print(deviation_metric(target_ones['per_square_meter_price'].values, test_ones_data['per_square_meter_price'].values))
    test_ones_data['id'] = id_data.loc[list(test_ones_data.index)]
    test_ones_data[['id', 'per_square_meter_price']].to_csv(args['o'], index=False)
    # print(test_ones_data['id'], target_ones['id'])

