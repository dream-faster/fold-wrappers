from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd
from fold.models.base import Model


class WrapStatsForecast(Model):
    properties = Model.Properties(
        model_type=Model.Properties.ModelType.regressor,
    )

    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        use_exogenous: bool,
        update_continuously: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        self.model = model_class(**init_args) if instance is None else instance
        self.model_class = model_class
        self.init_args = init_args
        self.use_exogenous = use_exogenous
        self.properties.requires_continuous_updates = update_continuously
        self.name = f"WrapStatsForecast-{self.model.__class__.__name__}"

    @classmethod
    def from_model(
        cls,
        model,
        use_exogenous: bool,
        update_continuously: bool = False,
    ) -> WrapStatsForecast:
        return cls(model_class=None, init_args=None, use_exogenous=use_exogenous, instance=model, update_continuously=update_continuously)  # type: ignore

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if self.use_exogenous:
            self.model.fit(y=y.values, X=X.values)
        else:
            self.model.fit(y=y.values)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if not hasattr(self.model, "forward"):
            return
        if self.use_exogenous:
            self.model.forward(y=y.values, h=len(X), X=X.values)
        else:
            self.model.forward(y=y.values, h=len(X))

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        if self.use_exogenous:
            return pd.Series(
                self.model.predict(h=len(X), X=X.values)["mean"], index=X.index
            )
        else:
            return pd.Series(self.model.predict(h=len(X))["mean"], index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pred_dict = self.model.predict_in_sample()
        if "fitted" in pred_dict:
            return pd.Series(pred_dict["fitted"], index=X.index)
        elif "mean" in pred_dict:
            return pd.Series(pred_dict["mean"], index=X.index)
        else:
            raise ValueError("Unknown prediction dictionary structure")
