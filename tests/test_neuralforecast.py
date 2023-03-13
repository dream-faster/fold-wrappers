import numpy as np
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data
from neuralforecast.models import NBEATS, NHITS

from fold_models.neuralforecast import WrapNeuralForecast


def test_neuralforecast_nbeats() -> None:
    X, y = generate_monotonous_data()

    step = 100
    input_size = 100
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=step)

    transformations = WrapNeuralForecast.from_model(
        NBEATS(
            input_size=input_size,
            h=step,
            stack_types=["identity", "trend"],
            max_epochs=50,
        ),
    )
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.2).all()


def test_neuralforecast_nhits() -> None:
    X, y = generate_monotonous_data()

    step = 100
    input_size = 100
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=step)

    transformations = WrapNeuralForecast.from_model(
        NHITS(
            input_size=input_size,
            h=step,
            max_epochs=50,
        ),
    )
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.2).all()
