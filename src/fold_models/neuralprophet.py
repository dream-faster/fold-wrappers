from copy import copy
from typing import Any, Optional, Union

import pandas as pd
from fold.models.base import Model


class WrapNeuralProphet(Model):
    properties = Model.Properties(
        model_type=Model.Properties.ModelType.regressor,
    )

    fitted = False

    def __init__(self, model: Any) -> None:
        self.model = model
        from neuralprophet import set_random_seed

        set_random_seed(0)
        self.name = "NeuralProphet"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        data = pd.DataFrame(
            {"ds": X.index, "y": y.values},
            index=range(0, len(y)),
        )
        self.training_metrics = self.model.fit(data, epochs=40)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        data = pd.DataFrame(
            {"ds": X.index, "y": 0.0},
            index=range(0, len(X)),
        )
        future = self.model.make_future_dataframe(data, periods=len(X))
        predictions = self.model.predict(future)["yhat1"]
        return pd.Series(
            predictions.values, index=X.index, name=self.name + "_predictions"
        )

    def __deepcopy__(self, memo):
        model = copy(self.model)
        setattr(model, "trainer", None)
        from io import BytesIO

        from torch import load, save

        buff = BytesIO()

        save(model, buff)
        buff.seek(0)
        model = load(buff)
        model.restore_trainer()
        return NeuralProphetWrapper(model)
