import catboost
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from mlens.ensemble import SuperLearner
from metrics import deviation_metric


class DataModel:
    def __init__(self, data, preproc_method=None):

        if preproc_method is not None:
            target = data['per_square_meter_price']
            price_types = data['price_type']
            data = preproc_method(data)
            data['price_type'] = price_types.values
            data['per_square_meter_price'] = target

        experts_data = data[data['price_type'] == 1]
        default_data = data[data['price_type'] == 0]

        self.e_X_train, self.e_X_test, self.e_y_train, self.e_y_test = train_test_split(
            experts_data.drop(columns=['price_type',
                                       'per_square_meter_price']),
            experts_data['per_square_meter_price'],
            test_size=0.33,
            random_state=42
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            default_data.drop(columns=['price_type',
                                       'per_square_meter_price']),
            default_data['per_square_meter_price'],
            test_size=0.33,
            random_state=42
        )


class HouseModel:

    def __init__(self):
        self.regression_model = self.__create_model()
        self.error_model = self.__create_error_model()
        self.error = 0

    def fit(self, data_model: DataModel):

        self.regression_model.fit(
            data_model.X_train,
            data_model.y_train
        )

        default_prediction = self.regression_model.predict(data_model.X_test)
        print(f'метрика предсказании пользовательских оценок {deviation_metric(data_model.y_test.values, default_prediction)}')

        expert_prediction = self.regression_model.predict(data_model.e_X_train)
        self.error = mean_absolute_error(data_model.e_y_train, expert_prediction)

        print(f'средняя ошибка при предсказании экспертных значений {self.error}')

        self.error_model.fit(
            data_model.e_X_train,
            expert_prediction - data_model.e_y_train
        )
    def predict(self, X):
        self.error = self.error_model.predict(X)
        return self.regression_model.predict(X) - self.error

    def __create_model(self):
        model = SuperLearner(verbose=2, random_state=21)
        model.add([
            KNeighborsRegressor(n_neighbors=10),
            RandomForestRegressor(random_state=21)
        ])

        model.add_meta(LinearRegression())
        return model

    def __create_error_model(self):
        model = catboost.CatBoostRegressor(iterations=10000, random_state=21)
        return model

if __name__ == '__main__':
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    ml_model = HouseModel()
    ml_model.fit(X_train, y_train)
    print(r2_score(y_test, ml_model.predict(X_test)))