from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from xgboost import XGBRegressor

from fold_models.xgboost import WrapXGB


def test_xgboost() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB.from_model(XGBRegressor())
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20


def test_xgboost_init_with_args() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB(XGBRegressor, {"n_estimators": 100})
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
