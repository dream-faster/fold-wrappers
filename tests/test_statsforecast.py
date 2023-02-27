from drift.loop import backtest, train
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_zeros_and_ones_skewed
from statsforecast.models import Naive

from drift_models.statsforecast_univariate import UnivariateStatsForecast


def test_statsforecast_univariate_model() -> None:

    X = generate_zeros_and_ones_skewed(70, weights=[0.5, 0.5])
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=50, step=1)
    transformations = UnivariateStatsForecast(Naive())

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        X.squeeze()[pred.index][:-2] == pred.shift(-2).squeeze()[:-2].astype(int)
    ).all()
