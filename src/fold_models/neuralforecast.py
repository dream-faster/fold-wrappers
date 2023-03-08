from __future__ import annotations

from typing import Any, Optional, Type, Union

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
        use_exogenous: bool,
        online_mode: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
        self.init_args = init_args
        self.use_exogenous = use_exogenous
        self.properties.mode = (
            Model.Properties.Mode.online
            if online_mode
            else Model.Properties.Mode.minibatch
        )
        self.name = f"WrapNeural-{self.model.__class__.__name__}"
        from neuralforecast import NeuralForecast

        self.nf = NeuralForecast(models=[self.model], freq="S")

    @classmethod
    def from_model(
        cls,
        model,
        use_exogenous: bool,
        online_mode: bool = False,
    ) -> WrapNeuralForecast:
        return cls(
            model_class=None,
            init_args=None,
            use_exogenous=use_exogenous,
            instance=model,
            online_mode=online_mode,
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

    def predict(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Union[pd.Series, pd.DataFrame]:
        predicted = self.nf.predict()

        if len(predicted) != len(X):
            raise ValueError("Step size and forecasting horizon must be equal.")
        else:
            return pd.Series(self.nf.predict(), index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return X
