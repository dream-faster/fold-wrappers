from __future__ import annotations

from typing import Any, Optional, Type, Union

import numpy as np
import pandas as pd
from fold.models.base import Model


class WrapNeuralForecast(Model):
    properties = Model.Properties(
        model_type=Model.Properties.ModelType.regressor,
    )

    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        instance: Optional[Any] = None,
    ) -> None:
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
        self.init_args = init_args
        self.name = f"WrapNeuraForecast-{self.model.__class__.__name__}"
        from neuralforecast import NeuralForecast

        self.nf = NeuralForecast(models=[self.model], freq="S")
        assert type(self.model.h) is int, "Forecasting horizon/step must be an integer."

    @classmethod
    def from_model(
        cls,
        model,
    ) -> WrapNeuralForecast:
        return cls(
            model_class=None,
            init_args=None,
            instance=model,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        data = pd.DataFrame(
            {"ds": X.index, "y": y.values, "unique_id": 1.0},
            index=range(0, len(y)),
        )
        self.nf.fit(data)

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        for model in self.nf.models:
            model.max_steps = 10
        data = pd.DataFrame(
            {"ds": X.index, "y": y.values, "unique_id": 1.0},
            index=range(0, len(y)),
        )
        self.nf.fit(data)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        predicted = self.nf.predict()

        if len(predicted) != len(X):
            raise ValueError(
                "Step size (of the Splitter) and `h` (forecasting horizon) must be equal."
            )
        else:
            return pd.Series(
                predicted[self.model.__class__.__name__].values, index=X.index
            )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = pd.DataFrame(
            {"ds": X.index, "y": [0.0] * len(X), "unique_id": 1.0},
            index=range(0, len(X)),
        )
        predictions = self.nf.predict_rolled(
            data,
            n_windows=int((len(X) - self.model.input_size) / self.model.h),
            step_size=self.model.h,
        )[self.model.__class__.__name__]
        # NeuralForecast will not return in sample predictions for `input_size`, so let's pad that with NaNs
        padding_size = len(X) - len(predictions)
        return pd.Series(
            np.hstack([np.full(padding_size, np.nan), predictions.values]),
            index=X.index,
        )
