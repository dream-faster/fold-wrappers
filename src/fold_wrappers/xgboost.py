from __future__ import annotations

from typing import Any, Callable, Optional, Type, Union

import pandas as pd
from fold.base import Tunable
from fold.models.base import Model


class WrapXGB(Model, Tunable):
    properties = Model.Properties(requires_X=True)

    def __init__(
        self,
        model_class: Type,
        init_args: Optional[dict] = {},
        instance: Optional[Any] = None,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.init_args = init_args
        self.model_class = model_class

        self.model = model_class(**init_args) if instance is None else instance
        from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

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
        self.name = name or self.model.__class__.__name__
        self.params_to_try = params_to_try

    @classmethod
    def from_model(
        cls,
        model,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> WrapXGB:
        return WrapXGB(
            model.__class__,
            init_args=model.get_params(),
            instance=model,
            name=name,
            params_to_try=params_to_try,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.fit(X=X, y=y, sample_weight=sample_weights)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.fit(
            X=X,
            y=y,
            xgb_model=self.model.get_booster(),
            sample_weight=sample_weights,
        )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(self.model.predict(X), index=X.index).rename(
            f"predictions_{self.name}"
        )
        if self.properties.model_type == Model.Properties.ModelType.classifier:
            probabilities = pd.DataFrame(
                data=self.model.predict_proba(X),
                index=X.index,
                columns=[
                    f"probabilities_{self.name}_{item}" for item in self.model.classes_
                ],
            )
            return pd.concat([predictions, probabilities], axis="columns")
        else:
            return predictions

    predict_in_sample = predict

    def get_params(self) -> dict:
        return self.model.get_params()

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapXGB(
            model_class=self.model_class,
            init_args=parameters,
            name=self.name,
        )
