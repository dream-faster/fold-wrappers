from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.tests import generate_monotonous_data
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.naive import NaiveForecaster
from utils import run_pipeline_and_check_if_results_close_univariate

from fold_wrappers.sktime import WrapSktime


def test_sktime_univariate_naiveforecaster() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapSktime(model_class=NaiveForecaster, init_args={}),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


def test_sktime_univariate_naiveforecaster_online() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapSktime(
            model_class=NaiveForecaster,
            init_args={},
            online_mode=True,
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )


def test_sktime_univariate_arima() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapSktime(
            model_class=ARIMA,
            init_args={"order": (1, 1, 0)},
            online_mode=False,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


def test_automatic_wrapping_sktime() -> None:
    X, y = generate_monotonous_data()
    train_backtest(
        ARIMA(order=(1, 1, 0)),
        X,
        y,
        splitter=SingleWindowSplitter(0.5),
    )


def test_sktime_univariate_arima_online() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapSktime(
            model_class=ARIMA,
            init_args={"order": (1, 1, 0)},
            online_mode=True,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


def test_sktime_univariate_autoarima() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapSktime(
            model_class=AutoARIMA,
            init_args={},
            online_mode=False,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


# def test_sktime_multivariate_autoarima() -> None:
#     run_pipeline_and_check_if_results_close_exogenous(
#         model=WrapSktime(
#             model_class=AutoARIMA,
#             init_args={},
#             use_exogenous=True,
#             online_mode=False,
#         ),
#         splitter=SingleWindowSplitter(train_window=50),
#     )


def test_sktime_univariate_autoarima_online() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapSktime(
            model_class=AutoARIMA,
            init_args={},
            online_mode=True,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )
