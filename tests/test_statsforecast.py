import numpy as np
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data
from statsforecast.models import ARIMA, MSTL, AutoARIMA, Naive
from utils import (
    run_pipeline_and_check_if_results_close_exogenous,
    run_pipeline_and_check_if_results_close_univariate,
)

from fold_models.statsforecast import WrapStatsForecast


def test_statsforecast_univariate_naive() -> None:
    X, y = generate_monotonous_data(length=70)

    splitter = ExpandingWindowSplitter(initial_train_window=50, step=1)
    pipeline = WrapStatsForecast(model_class=Naive, init_args={})
    trained_pipelines = train(pipeline, None, y, splitter)
    pred = backtest(trained_pipelines, None, y, splitter)
    assert np.isclose(
        y.squeeze().shift(1)[pred.index][:-1], pred.squeeze()[:-1], atol=0.01
    ).all()


def test_statsforecast_univariate_autoarima() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsForecast.from_model(AutoARIMA()),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=5),
    )


def test_statsforecast_exogenous_autoarima() -> None:
    run_pipeline_and_check_if_results_close_exogenous(
        model=[WrapStatsForecast.from_model(AutoARIMA())],
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=5),
    )


def test_statsforecast_univariate_arima() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsForecast(model_class=ARIMA, init_args={"order": (1, 0, 0)}),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=2),
    )


# def test_statsforecast_univariate_arima_online() -> None:
#     run_pipeline_and_check_if_results_are_close(
#         model=WrapStatsForecast(
#             model_class=ARIMA,
#             init_args={"order": (1, 0, 0)},
#             online_mode=True,
#         ),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )


def test_statsforecast_univariate_mstl() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsForecast.from_model(MSTL(season_length=10), online_mode=False),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )


# def test_statsforecast_univariate_mstl_online() -> None:
#     run_pipeline_and_check_if_results_are_close(
#         model=WrapStatsForecast.from_model(
#             MSTL(season_length=10), online_mode=False
#         ),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )
