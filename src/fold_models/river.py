from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd
from fold.models.base import Model


class WrapRiver(Model):
    properties = Model.Properties(
        model_type=Model.Properties.ModelType.regressor,
    )

    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        online_mode: bool = True,
        instance: Optional[Any] = None,
    ) -> None:
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
        self.init_args = init_args
        self.properties.mode = (
            Model.Properties.Mode.online
            if online_mode
            else Model.Properties.Mode.minibatch
        )
        self.name = f"WrapRiver-{self.model.__class__.__name__}"

    @classmethod
    def from_model(
        cls,
        model,
        online_mode: bool = True,
    ) -> WrapRiver:
        return cls(
            model_class=None,
            init_args=None,
            instance=model,
            online_mode=online_mode,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.learn_many(X, y)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        self.model.predict()

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        self.model.predict_in_sample()
