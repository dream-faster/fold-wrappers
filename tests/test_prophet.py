from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data
from prophet import Prophet
import numpy as np
from fold_models.prophet import WrapProphet


def test_prophet() -> None:
    X, y = generate_sine_wave_data(cycles=100, length=2400, freq="H")

    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.1)
    transformations = WrapProphet.from_model(Prophet())

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.1).all()


test_prophet()
