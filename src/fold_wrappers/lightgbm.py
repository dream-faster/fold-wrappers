from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

import pandas as pd
from fold.models.base import Model


class WrapLGBM(Model):
    properties = Model.Properties(requires_X=True)

    def __init__(
        self,
        model_class: Type,
        init_args: Optional[Dict],
        instance: Optional[Any] = None,
    ) -> None:
        self.init_args = init_args
        init_args = {} if init_args is None else init_args
        self.model_class = model_class

        self.model = model_class(**init_args) if instance is None else instance
        from lightgbm import LGBMClassifier, LGBMRegressor

        if isinstance(self.model, LGBMRegressor):
            self.properties.model_type = Model.Properties.ModelType.regressor
        elif isinstance(self.model, LGBMClassifier):
            self.properties.model_type = Model.Properties.ModelType.classifier
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
        self.name = self.model_class.__class__.__name__

    @classmethod
    def from_model(
        cls,
        model,
    ) -> WrapLGBM:
        return WrapLGBM(model.__class__, {}, instance=model)

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
            init_model=self.model.get_booster(),
        )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(self.model.predict(X), index=X.index)
