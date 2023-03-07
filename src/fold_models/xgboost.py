from typing import Any, Optional

import pandas as pd
from fold.models.base import Model


class XGB(Model):
    properties = Model.Properties()

    fitted = False

    def __init__(self, model: Any) -> None:
        from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

        self.model = model
        self.name = "XGB-{model.__class__.__name__}"
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

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if self.fitted:
            self.model.fit(
                X,
                y,
                sample_weight=sample_weights,
                xgb_model=self.model.get_booster(),
            )
        else:
            self.model.fit(X, y, sample_weight=sample_weights)
            self.fitted = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.Series(self.model.predict(X), index=X.index)
