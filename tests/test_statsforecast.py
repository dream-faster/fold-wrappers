import numpy as np
from drift.loop import backtest, train
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_sine_wave_data
from statsforecast.models import Naive

from drift_models.statsforecast_univariate import UnivariateStatsForecast


def test_statsforecast_univariate_model() -> None:

    X = generate_sine_wave_data()
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = UnivariateStatsForecast(Naive())

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, pred.values))


test_statsforecast_univariate_model()
