from typing import Optional, Union

import pandas as pd
from fold.models.base import Model
from sklearn.linear_model import LinearRegression


class AR(Model):
    def __init__(self, p: int) -> None:
        self.p = p
        self.name = f"AR-{str(p)}"
        self.properties = Model.Properties(
            mode=Model.Properties.Mode.online,
            model_type=Model.Properties.ModelType.regressor,
            memory_size=p,
            _internal_supports_minibatch_backtesting=True,
        )
        self.models = [LinearRegression() for _ in range(p)]

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        for index, lr in enumerate(self.models, start=1):
            lr.fit(y.shift(index), y)

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        for index, lr in enumerate(self.models, start=1):
            lr.partial_fit(y.shift(index), y)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self.lr.predict(self._state.memory_y.iloc[-1:None])

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self.lr.predict(self._state.memory_y.shift(1))


# class SAR(Model):
#     def __init__(self, p: int, season_length: int) -> None:
#         self.p = p
#         self.name = f"SAR-{str(p)}-{str(season_length)}"
#         self.models = [LinearRegression() for _ in range(season_length)]
#         self.season_length = season_length

#     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
#         seasonal_X_y = [
#             (
#                 X.shift(season).to_frame().values[:: self.season_length],
#                 y.shift(season).values[:: self.season_length],
#             )
#             for season in range(self.season_length)
#         ]
#         for index, local_X_y in enumerate(seasonal_X_y):
#             local_X = local_X_y[0]
#             local_X[0] = local_X[1]
#             local_y = local_X_y[1]
#             local_y[0] = local_y[1]
#             self.models[index].fit(local_X, local_X_y[1])
#         self.models = shift(self.models, len(X) % self.season_length)

#     def predict(self, X: pd.DataFrame) -> pd.Series:
#         preds = [
#             self.models[index % self.season_length].predict([[item]])
#             for index, item in enumerate(X)
#         ]
#         return pd.DataFrame(preds, index=X.index)
