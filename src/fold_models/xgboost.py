from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd
from fold.models.base import Model

from .base import Wrapper


class WrapXGB(Wrapper):
    properties = Model.Properties()

    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        instance: Optional[Any] = None,
    ) -> None:
        self.model_class = model_class
        self.init_args = init_args

        self.model = model_class(**init_args) if instance is None else instance
        from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

        self.name = f"XGB-{self.model.__class__.__name__}"
        if isinstance(self.model, XGBRegressor) or isinstance(
            self.model, XGBRFRegressor
        ):
            self.properties.model_type = Model.Properties.ModelType.regressor
        elif isinstance(self.model, XGBClassifier) or isinstance(
            self.model, XGBRFClassifier
        ):
            self.properties.model_type = Model.Properties.ModelType.classifier
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
        self.name = f"WrapXGB-{self.model_class.__class__.__name__}"

    @classmethod
    def from_model(
        cls,
        model,
    ) -> WrapXGB:
        return WrapXGB(model.__class__, {}, instance=model)

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.fit(X, y, sample_weight=sample_weights)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.fit(
            X,
            y,
            sample_weight=sample_weights,
            xgb_model=self.model.get_booster(),
        )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(self.model.predict(X), index=X.index)
