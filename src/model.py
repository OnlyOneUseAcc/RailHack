from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from mlens.ensemble import SuperLearner


class HouseModel:

    def __init__(self):
        self.regression_model = self.__create_model()

    def fit(self, X, y):
        self.regression_model.fit(X, y)

    def predict(self, X):
        return self.regression_model.predict(X)

    def __create_model(self):
        model = SuperLearner(verbose=2)
        model.add([
            KNeighborsRegressor(n_neighbors=10),
            RandomForestRegressor()
        ])

        model.add_meta( LinearRegression())
        return model

#class ErrorMin:


if __name__ == '__main__':
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    ml_model = HouseModel()
    ml_model.fit(X_train, y_train)
    print(r2_score(y_test, ml_model.predict(X_test)))
