from fold.splitters import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from utils import run_pipeline_and_check_if_results_are_close

from fold_models.sktime import WrapSktime


def test_sktime_univariate_naiveforecaster() -> None:
    run_pipeline_and_check_if_results_are_close(
        model=WrapSktime(
            model_class=NaiveForecaster, init_args={}, use_exogenous=False
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )
