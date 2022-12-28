from typing import Any

import numpy as np
from drift.models.base import Model, ModelType


class UnivariateStatsForecastModel(Model):

    name = "UnivariateStatsForecastModel"
    type = ModelType.Univariate

    def __init__(self, model: Any) -> None:
        self.model = model
        self.name = "UnivariateStatsForecast-{model.alias}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(y=X)

    def predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        pred_dict = self.model.predict_in_sample()
        if "fitted" in pred_dict:
            return pred_dict["fitted"]
        elif "mean" in pred_dict:
            return pred_dict["mean"]
        else:
            raise ValueError("Unknown prediction dictionary structure")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(h=len(X))["mean"]
