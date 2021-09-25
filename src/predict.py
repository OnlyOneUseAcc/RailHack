import pickle
import argparse
import pandas as pd


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

    with open(f'{args}', 'rb') as model_file:
        model = pickle.load(model_file)

    test_data['per_square_meter_price'] = model.predict(test_data)

    test_data[['id', 'per_square_meter_price']].to_csv(args['o'], index=False)
