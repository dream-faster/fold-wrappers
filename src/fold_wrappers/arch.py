from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from fold.models.base import Model
from fold.utils.checks import is_X_available


class WrapArch(Model):
    properties = Model.Properties(
        requires_X=False,
        model_type=Model.Properties.ModelType.regressor,
    )

    def __init__(
        self,
        init_args: dict,
        use_exogenous: Optional[bool] = None,
        online_mode: bool = False,
        instance: Optional[Any] = None,
    ) -> None:
        self.init_args = init_args
        init_args = {} if init_args is None else init_args
        self.use_exogenous = use_exogenous
        self.properties.mode = (
            Model.Properties.Mode.online
            if online_mode
            else Model.Properties.Mode.minibatch
        )
        self.name = "Arch"
        self.instance = instance

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        from arch import arch_model

        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model = arch_model(y, x=X, **self.init_args)
            self.model = self.model.fit()
        else:
            self.model = arch_model(y, **self.init_args)
            self.model = self.model.fit()

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if not hasattr(self.model, "append"):
            return
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            self.model = self.model.fit(starting_values=self.model.params, last_obs=y)
        else:
            self.model = self.model.append(endog=y, refit=True)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        use_exogenous = (
            is_X_available(X) if self.use_exogenous is None else self.use_exogenous
        )
        if use_exogenous:
            return pd.Series(self.model.forecast(horizon=len(X), x=X))
        else:
            return pd.Series(self.model.forecast(horizon=len(X)))

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        res = self.model.forecast(horizon=len(X), start=0, reindex=True)
        return res.variance[res.variance.columns[0]]
