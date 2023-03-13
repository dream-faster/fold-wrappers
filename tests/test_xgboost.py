from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from xgboost import XGBRegressor

from fold_models.xgboost import WrapXGB


def test_xgboost() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB(XGBRegressor())

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
