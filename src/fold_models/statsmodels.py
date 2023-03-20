from __future__ import annotations

from typing import Any, Optional, Type, Union

import pandas as pd
from fold.models.base import Model

from .base import Wrapper


class WrapStatsModels(Wrapper):
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
        self.model_class = model_class
        self.init_args = init_args
        self.use_exogenous = use_exogenous
        self.properties.mode = (
            Model.Properties.Mode.online
            if online_mode
            else Model.Properties.Mode.minibatch
        )
        self.name = f"WrapStatsModels-{self.model_class.__class__.__name__}"
        self.instance = instance

    @classmethod
    def from_model(
        cls,
        model,
        use_exogenous: bool,
        online_mode: bool = False,
    ) -> WrapStatsModels:
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
        if self.use_exogenous:
            self.model = (
                self.model_class(y, X, **self.init_args)
                if self.instance is None
                else self.instance
            )
            self.model = self.model.fit()
        else:
            self.model = (
                self.model_class(y, **self.init_args)
                if self.instance is None
                else self.instance
            )
            self.model = self.model.fit()

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if not hasattr(self.model, "append"):
            return

        if self.use_exogenous:
            self.model = self.model.append(endog=y, exog=X, refit=True)
        else:
            self.model = self.model.append(endog=y, refit=True)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        if self.use_exogenous:
            return pd.Series(
                self.model.predict(start=X.index[0], end=X.index[-1], exog=X)
            )
        else:
            return pd.Series(self.model.predict(start=X.index[0], end=X.index[-1]))

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self.model.predict(start=X.index[0], end=X.index[-1])
