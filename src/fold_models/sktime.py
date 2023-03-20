from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd
from fold.models.base import Model
from fold.utils.checks import is_X_available
from sktime.forecasting.base import ForecastingHorizon


class WrapSktime(Model):
    properties = Model.Properties(
        model_type=Model.Properties.ModelType.regressor,
    )

    def __init__(
        self,
        model_class: Type,
        init_args: Optional[dict],
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        init_args = {} if init_args is None else init_args
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
        self.init_args = init_args
        self.use_exogenous = use_exogenous
        self.properties.mode = (
            Model.Properties.Mode.online
            if online_mode
            else Model.Properties.Mode.minibatch
        )
        self.name = f"WrapStatsForecast-{self.model.__class__.__name__}"

    @classmethod
    def from_model(
        cls,
        model,
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
    ) -> WrapSktime:
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
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model.fit(y=y, X=X)
        else:
            self.model.fit(y=y)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if not hasattr(self.model, "update"):
            return
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model = self.model.update(y=y, X=X, update_params=True)
        else:
            self.model = self.model.update(y=y, update_params=True)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        fh = ForecastingHorizon(list(range(1, len(X) + 1)), is_relative=True)
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            return self.model.predict(fh, X=X)
        else:
            return self.model.predict(fh)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        fh = ForecastingHorizon(X.index, is_relative=False)
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            return self.model.predict(fh, X=X)
        else:
            return self.model.predict(fh)
