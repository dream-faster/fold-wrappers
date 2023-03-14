from fold.splitters import ExpandingWindowSplitter, Splitter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils import run_pipeline_and_check_if_results_are_close

from fold_models.statsmodels import WrapStatsModels


def test_statsmodels_univariate_arima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsModels(
            model_class=ARIMA, init_args={"order": (1, 0, 0)}, use_exogenous=False
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


# def test_statsforecast_univariate_arima_online() -> None:
#     run_pipeline_and_check_if_results_are_close(
#         model=WrapStatsModels(
#             model_class=ARIMA,
#             init_args={"order": (1, 0, 0)},
#             use_exogenous=False,
#             online_mode=True,
#         ),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )


# def test_statsforecast_multivariate_arima_online() -> None:
#     run_pipeline_and_check_if_results_are_close(
#         model=WrapStatsModels(
#             model_class=ARIMA,
#             init_args={"order": (1, 0, 0)},
#             use_exogenous=True,
#             online_mode=True,
#         ),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )


def test_statsmodels_multivariate_arima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsModels(
            model_class=ARIMA, init_args={"order": (1, 0, 0)}, use_exogenous=True
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


def test_statsmodels_univariate_exponential_smoothing() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsModels(
            model_class=ExponentialSmoothing,
            init_args={},
            use_exogenous=False,
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )
