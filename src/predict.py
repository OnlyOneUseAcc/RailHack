import argparse
import pandas as pd

from utils import default_preprocess
from settings import NUM_FEATURES
from mlens.utils.utils import pickle_load

def parse_args():
    parser = argparse.ArgumentParser(
        description="Скрипт для предсказания модели",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--test_data", "-d", type=str, dest="d", required=True,
                        help="Путь до отложенной выборки")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", default="models/model.pkl",
                        help="Пусть до сериализованной ML модели")
    parser.add_argument("--output", "-o", type=str, dest="o", default='data/test_submission.csv',
                        help="Путь до выходного файла")

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())

    test_data = pd.read_csv(args['d'])

    test_data.loc[:,  NUM_FEATURES] = default_preprocess(test_data)

    model = pickle_load(name=f'{args["mp"]}')
    test_data['per_square_meter_price'] = model.predict(test_data.loc[:,  NUM_FEATURES])
    test_data[['id', 'per_square_meter_price']].to_csv(args['o'], index=False)


