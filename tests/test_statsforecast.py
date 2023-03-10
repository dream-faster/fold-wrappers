import numpy as np
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data
from statsforecast.models import ARIMA, AutoARIMA, Naive

from fold_models.statsforecast import WrapStatsForecast


def test_statsforecast_univariate_naive() -> None:
    X, y = generate_monotonous_data(length=70)

    splitter = ExpandingWindowSplitter(train_window_size=50, step=1)

    transformations = WrapStatsForecast(
        model_class=Naive, init_args={}, use_exogenous=False
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(
        y.squeeze().shift(1)[pred.index][:-1], pred.squeeze().values[:-1], atol=0.01
    ).all()


def test_statsforecast_univariate_naive_continuous_update() -> None:
    X, y = generate_monotonous_data(length=70)

    splitter = ExpandingWindowSplitter(train_window_size=50, step=10)

    transformations = WrapStatsForecast(
        model_class=Naive,
        init_args={},
        use_exogenous=False,
        update_continuously=True,
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(
        y.squeeze().shift(1)[pred.index][:-1], pred.squeeze().values[:-1], atol=0.01
    ).all()


def test_statsforecast_univariate_autoarima() -> None:
    X, y = generate_monotonous_data(length=70)

    splitter = ExpandingWindowSplitter(train_window_size=50, step=1)

    transformations = transformations = WrapStatsForecast.from_model(
        AutoARIMA(), use_exogenous=False
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()


def test_statsforecast_univariate_arima() -> None:
    X, y = generate_monotonous_data(length=70)
    splitter = ExpandingWindowSplitter(train_window_size=50, step=1)

    transformations = transformations = WrapStatsForecast(
        model_class=ARIMA, init_args={"order": (1, 0, 0)}, use_exogenous=False
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()


def test_statsforecast_univariate_arima_continuous() -> None:
    X, y = generate_monotonous_data(length=70)
    splitter = ExpandingWindowSplitter(train_window_size=50, step=10)

    transformations = transformations = WrapStatsForecast(
        model_class=ARIMA,
        init_args={"order": (1, 0, 0)},
        use_exogenous=False,
        update_continuously=True,
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()


# def test_statsforecast_univariate_mstl() -> None:
#     X, y = generate_monotonous_data(length=70)
#     splitter = ExpandingWindowSplitter(train_window_size=50, step=1)

#     transformations = transformations = WrapStatsForecast.from_model(
#         MSTL(season_length=10), use_exogenous=False, update_continuously=True
#     )

#     transformations_over_time = train(transformations, X, y, splitter)
#     pred = backtest(transformations_over_time, X, y, splitter)
#     assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()
