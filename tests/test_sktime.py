from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.naive import NaiveForecaster
from utils import run_pipeline_and_check_if_results_are_close

from fold_models.sktime import WrapSktime


def test_sktime_univariate_naiveforecaster() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(model_class=NaiveForecaster, init_args={}),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


def test_sktime_univariate_naiveforecaster_online() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=NaiveForecaster,
            init_args={},
            use_exogenous=False,
            online_mode=True,
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )


def test_sktime_univariate_arima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=ARIMA,
            init_args={"order": (1, 1, 0)},
            use_exogenous=False,
            online_mode=False,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


def test_sktime_univariate_arima_online() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=ARIMA,
            init_args={"order": (1, 1, 0)},
            use_exogenous=False,
            online_mode=True,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


def test_sktime_univariate_autoarima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=AutoARIMA,
            init_args={},
            use_exogenous=False,
            online_mode=False,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


def test_sktime_multivariate_autoarima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=AutoARIMA,
            init_args={},
            use_exogenous=True,
            online_mode=False,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )


def test_sktime_univariate_autoarima_online() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=AutoARIMA,
            init_args={},
            use_exogenous=False,
            online_mode=True,
        ),
        splitter=SingleWindowSplitter(train_window=50),
    )
