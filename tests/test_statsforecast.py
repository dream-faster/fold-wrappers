import pytest
from drift.loop import walk_forward_inference, walk_forward_train
from drift.utils.splitters import ExpandingWindowSplitter
from statsforecast.models import Theta

from drift_models.statsforecast_univariate import UnivariateStatsForecastModel
from tests.utils import generate_sine_wave_data


def test_statsforecast_univariate_model() -> None:

    X = generate_sine_wave_data()
    y = X

    splitter = ExpandingWindowSplitter(start=0, end=len(y), window_size=400, step=400)
    model = UnivariateStatsForecastModel(Theta())

    model_over_time = walk_forward_train(model, X, y, splitter, None)

    _, pred = walk_forward_inference(model_over_time, None, X, y, splitter)
