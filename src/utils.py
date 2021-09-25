import pandas as pd
from impyute.imputation.cs import mice
from tqdm import tqdm
from baseline.raifhack_ds.settings import NUM_FEATURES
import numpy as np
import re

THRESHOLD = 7500
THRESHOLD_CAPITAL = 3000


def fill_empty_values(data: pd.DataFrame):
    imputed_training = mice(data.values)
    empty_mask = data.isna()
    data_array = data.values
    data_array[empty_mask] = imputed_training[empty_mask]
    return pd.DataFrame(data_array,
                        columns=data.columns,
                        index=data.index)


def fill_empty_values_by_location(full_data):
    city_table = full_data[['city', 'id']].groupby(by=['city']).count()

    for unique_region in pd.unique(full_data.loc[:, 'region']):
        region_data = full_data[full_data['region'] == unique_region]
        print(unique_region)
        for unique_city in tqdm(pd.unique(region_data.loc[:, 'city'])):
            current_city_table = region_data[region_data['city'] == unique_city].copy()
            city_indexes = current_city_table.index

            if city_table.loc[unique_city, 'id'] > THRESHOLD:
                current_city_table.loc[:, NUM_FEATURES] = fill_empty_values(current_city_table[NUM_FEATURES])
                full_data.loc[city_indexes, :] = current_city_table
            else:
                city_len = region_data[['id', 'city']].groupby(by='city').count()

                if city_len[city_len.id > THRESHOLD_CAPITAL].shape[0] > 0:
                    names = list(city_len[city_len['id'] < THRESHOLD_CAPITAL].index)
                    current_region_data = region_data[~ region_data['city'].isin(names)]

                    full_region_data = fill_empty_values(current_region_data.loc[:, NUM_FEATURES])
                    full_data.loc[city_indexes, NUM_FEATURES] = full_region_data.loc[city_indexes, NUM_FEATURES]
                else:
                    full_region_data = fill_empty_values(region_data.loc[:, NUM_FEATURES])
                    full_data.loc[city_indexes, NUM_FEATURES] = full_region_data.loc[city_indexes, NUM_FEATURES]
    return full_data


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


def drop_corr(data: pd.DataFrame, tresh=0.9):
    data_corr = data.corr()

    for i in range(data_corr.shape[0]):
        data_corr.iloc[i, i + 1:] = 0
    drop_columns = list()

    for col in data_corr.columns:
        corr_values = data_corr.loc[(data_corr[col] >= tresh) & (data_corr[col] != 1)]
        if col not in drop_columns and len(corr_values.index) != 0:
            drop_columns.extend(corr_values.index)

    return drop_columns
