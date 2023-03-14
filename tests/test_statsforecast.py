import numpy as np
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter, Splitter
from fold.utils.tests import generate_monotonous_data
from statsforecast.models import ARIMA, MSTL, AutoARIMA, Naive
from utils import run_pipeline_and_check_if_results_are_close

from fold_models.statsforecast import WrapStatsForecast


def test_statsforecast_univariate_naive() -> None:
    X, y = generate_monotonous_data(length=70)

    splitter = ExpandingWindowSplitter(initial_train_window=50, step=1)

    transformations = WrapStatsForecast(
        model_class=Naive, init_args={}, use_exogenous=False
    )

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(
        y.squeeze().shift(1)[pred.index][:-1], pred.squeeze().values[:-1], atol=0.01
    ).all()


def test_statsforecast_univariate_autoarima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsForecast.from_model(AutoARIMA(), use_exogenous=False),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


def test_statsforecast_univariate_arima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsForecast(
            model_class=ARIMA, init_args={"order": (1, 0, 0)}, use_exogenous=False
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


# def test_statsforecast_univariate_arima_online() -> None:
#     run_pipeline_and_check_if_results_are_close(
#         model=WrapStatsForecast(
#             model_class=ARIMA,
#             init_args={"order": (1, 0, 0)},
#             use_exogenous=False,
#             online_mode=True,
#         ),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )


def test_statsforecast_univariate_mstl() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsForecast.from_model(
            MSTL(season_length=10), use_exogenous=False, online_mode=False
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )


# def test_statsforecast_univariate_mstl_online() -> None:
#     run_pipeline_and_check_if_results_are_close(
#         model=WrapStatsForecast.from_model(
#             MSTL(season_length=10), use_exogenous=False, online_mode=False
#         ),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )
