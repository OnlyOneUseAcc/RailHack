import pickle
import pandas as pd
from tqdm import tqdm
from src.settings import NUM_FEATURES
from sklearn.ensemble import IsolationForest
import numpy as np
import re
from sklearn import preprocessing

THRESHOLD = 7500
THRESHOLD_CAPITAL = 3000

DROPPED_COLUMNS = ['city', 'osm_city_nearest_name', 'region', 'street', 'date', 'price_type', 'floor']


def get_floor_info(data: pd.DataFrame, n=5):
    popular_words = get_popular_words(data, n)
    data[popular_words] = 0
    data = fill_popular_words(data, popular_words)
    range_floors = {5: '-5-5', 15: '6-15', 25: '16-25', 50: '26-50', 100: '51-100', 10000: '101+'}
    data[list(range_floors.values())] = 0
    data = fill_floors(data, range_floors)

    return data


def get_popular_words(data: pd.DataFrame, n=5):
    popular_words = list()
    for line in data.loc[data.floor.notna(), 'floor']:
        popular_words.extend(re.findall('[а-яА-Я]{3,}', str(line).lower()))

    popular_words = np.unique(popular_words, return_counts=True)

    popular_words = pd.DataFrame(index=popular_words[0], data=popular_words[1]) \
        .sort_values(by=0, ascending=False).head().index.values

    return popular_words


def fill_popular_words(data: pd.DataFrame, popular_words):
    data[popular_words] = 0
    for ind in data.index.values:
        for popular_word in popular_words:
            if popular_word in str(data.loc[ind, 'floor']):
                data.loc[ind, popular_word] = 1
    return data


def fill_floors(data: pd.DataFrame, range_floors):
    for ind in tqdm(data.index.values):
        line = str(data.loc[ind, 'floor'])
        floor_range = re.findall('[0-9]+-[0-9]+', line)
        floors = re.findall('-?[0-9]*[.]?[0-9]+', line)
        if len(floor_range) != 0:
            for fl in floor_range:
                for floor in range(int(fl[0]), int(fl[-1])):
                    for key in list(range_floors.keys()):
                        if floor <= key:
                            data.loc[ind, range_floors[key]] = +1
                            break
        elif len(floors) != 0:
            for fl in floors:
                for key in list(range_floors.keys()):
                    if float(fl) <= key:
                        data.loc[ind, range_floors[key]] = +1
                        break

    return data


def drop_corr(data: pd.DataFrame, tresh=0.8):
    data_corr = data.corr()

    for i in range(data_corr.shape[0]):
        data_corr.iloc[i, i + 1:] = 0
    drop_columns = list()

    for col in data_corr.columns:
        corr_values = data_corr.loc[(data_corr[col] >= tresh) & (data_corr[col] != 1)]
        if col not in drop_columns and len(corr_values.index) != 0:
            drop_columns.extend(corr_values.index)

    return set(drop_columns)


def dropna_value(data):
    drop_cols = ['osm_city_nearest_population', 'street']
    data.drop(columns=drop_cols, inplace=True)
    corr_data = data.dropna()
    return corr_data


def drop_anomaly(data):
    clf = IsolationForest(max_samples=100, random_state=1, contamination='auto')
    preds = clf.fit_predict(data.drop(columns=['city', 'id', 'region', 'date', 'osm_city_nearest_name']))
    data['anomaly'] = preds
    print(data.shape, len(preds))
    print(np.unique(preds))
    data = data.loc[data.anomaly == 1]
    print(data.shape, sum(preds))
    data.drop(columns='anomaly', inplace=True)
    return data


def drop_price(data, target, treshhold_full_price=2 + 1e8):
    corr_data = data.loc[(data['total_square'] * data[target]) < (treshhold_full_price)]
    return corr_data


CORR_COLS = ['osm_amenity_points_in_0.01',
             'osm_healthcare_points_in_0.01',
             'osm_hotels_points_in_0.01',
             'osm_offices_points_in_0.01',
             'osm_shops_points_in_0.001',
             'reform_mean_floor_count_500',
             'osm_shops_points_in_0.01',
             'osm_crossing_points_in_0.01',
             'osm_hotels_points_in_0.0075',
             'osm_healthcare_points_in_0.0075',
             'osm_crossing_points_in_0.0075',
             'osm_culture_points_in_0.01',
             'osm_historic_points_in_0.0075',
             'osm_finance_points_in_0.0075',
             'reform_mean_year_building_500',
             'osm_historic_points_in_0.01',
             'osm_building_points_in_0.0075',
             'osm_catering_points_in_0.005',
             'reform_count_of_houses_500',
             'osm_offices_points_in_0.0075',
             'osm_catering_points_in_0.01',
             'osm_culture_points_in_0.0075',
             'osm_shops_points_in_0.005',
             'osm_amenity_points_in_0.0075',
             'osm_transport_stop_points_in_0.0075',
             'osm_finance_points_in_0.01',
             'osm_leisure_points_in_0.0075',
             'osm_shops_points_in_0.0075',
             'osm_catering_points_in_0.0075']


def default_preprocess(data):
    data = data.drop(columns=CORR_COLS)

    data.loc[:, 'date'] = pd.to_datetime(data.date, format='%Y-%m-%d')
    data.loc[:, 'month'] = data['date'].dt.month
    data.loc[:, 'day'] = data['date'].dt.day

    data = data.drop(columns=DROPPED_COLUMNS)
    data = data.apply(lambda col: col.fillna(col.mean()), axis=0)

    return data


def train_preproc(data):
    target = 'per_square_meter_price'
    data = drop_price(data, target)
    data = data.drop(columns='id')
    data.drop(columns=target, inplace=True)

    data = default_preprocess(data)

    with open('../models/col_list.pkl', 'wb') as list_columns:
        train_columns = data.columns
        pickle.dump(train_columns, list_columns)

    scaler = preprocessing.MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data),
                        columns=data.columns,
                        index=data.index)
    with open('../models/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return data


def test_preproc(data, scaler):
    data = data.drop(columns='id')
    data = default_preprocess(data)

    with open('../models/col_list.pkl', 'rb') as list_columns:
        train_columns = pickle.load(list_columns)
        data = data.loc[:, train_columns]

    data = pd.DataFrame(scaler.transform(data),
                        columns=data.columns,
                        index=data.index)

    return data
