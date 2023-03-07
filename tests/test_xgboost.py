from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from xgboost import XGBRegressor

from fold_models.xgboost import XGB


def test_xgboost() -> None:
    X = generate_sine_wave_data()
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=50, step=10)
    transformations = XGB(XGBRegressor())

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (y.squeeze()[pred.index][:-1] - pred.squeeze()[:-1]).abs().sum() < 20
