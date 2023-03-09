from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from statsforecast.models import ARIMA, Naive

from fold_models.statsforecast import WrapStatsForecast


def test_statsforecast_univariate_naive() -> None:
    X = generate_sine_wave_data(resolution=70)
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=50, step=1)

    transformations = WrapStatsForecast(
        model_class=Naive, init_args={}, use_exogenous=False
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        y.squeeze()[pred.index][:-1] == pred.shift(-1).squeeze()[:-1].astype(int)
    ).all()


def test_statsforecast_univariate_arima() -> None:
    X = generate_sine_wave_data(resolution=70)
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=50, step=1)

    transformations = transformations = WrapStatsForecast(
        model_class=ARIMA, init_args={"order": (1, 0, 0)}, use_exogenous=False
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (y.squeeze()[pred.index][:-1] == pred.squeeze()[:-1].astype(int)).all()
