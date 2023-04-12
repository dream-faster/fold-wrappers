from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from lightgbm import LGBMRegressor

from fold_models.lightgbm import WrapLGBM


def test_xgboost() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM.from_model(LGBMRegressor())

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20


def test_xgboost_init_with_args() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM(LGBMRegressor, {"n_estimators": 100})

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
