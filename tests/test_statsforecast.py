from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_zeros_and_ones_skewed
from statsforecast.models import Naive

from fold_models.statsforecast import UnivariateStatsForecast


def test_statsforecast_univariate_model() -> None:

    X = generate_zeros_and_ones_skewed(70, weights=[0.5, 0.5])
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=50, step=10)
    transformations = UnivariateStatsForecast(Naive())

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        y.squeeze()[pred.index][:-1] == pred.shift(-1).squeeze()[:-1].astype(int)
    ).all()
