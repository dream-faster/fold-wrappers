import numpy as np
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter, Splitter
from fold.utils.tests import generate_monotonous_data
from statsmodels.tsa.arima.model import ARIMA

from fold_models.statsmodels import WrapStatsModels


def run_pipeline_and_check_if_results_are_close(model, splitter: Splitter):
    X, y = generate_monotonous_data(length=70)

    transformations_over_time = train(model, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()


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


def test_statsmodels_multivariate_arima() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapStatsModels(
            model_class=ARIMA, init_args={"order": (1, 0, 0)}, use_exogenous=True
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )
