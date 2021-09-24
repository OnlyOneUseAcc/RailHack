import pandas as pd
from impyute.imputation.cs import mice
from tqdm import tqdm
from baseline.raifhack_ds.settings import NUM_FEATURES


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




