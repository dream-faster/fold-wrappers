from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd
from fold.models.base import Model


class WrapProphet(Model):
    properties = Model.Properties(
        model_type=Model.Properties.ModelType.regressor,
    )

    def __init__(
        self,
        model_class: Type,
        init_args: Optional[dict],
        online_mode: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        init_args = {} if init_args is None else init_args
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
        self.init_args = init_args
        self.properties.mode = (
            Model.Properties.Mode.online
            if online_mode
            else Model.Properties.Mode.minibatch
        )
        self.name = f"WrapProphet-{self.model.__class__.__name__}"

    @classmethod
    def from_model(
        cls,
        model,
        online_mode: bool = False,
    ) -> WrapProphet:
        return cls(
            model_class=model.__class__,
            init_args={},
            instance=model,
            online_mode=online_mode,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        data = pd.DataFrame({"ds": X.index, "y": y.values})
        self.model.fit(data)

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = pd.DataFrame({"ds": X.index})
        predictions = self.model.predict(data)
        return pd.Series(predictions["yhat"].values, index=X.index)

    predict_in_sample = predict
