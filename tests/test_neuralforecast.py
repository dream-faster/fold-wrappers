import numpy as np
import pandas as pd
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, RNN

from fold_models.neuralforecast import WrapNeuralForecast


def test_neuralforecast_nbeats() -> None:
    X, y = generate_sine_wave_data(cycles=50)

    step = 100
    input_size = 100
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=step)

    transformations = WrapNeuralForecast.from_model(
        NBEATS(
            input_size=input_size,
            h=step,
            max_steps=50,
        ),
    )
    transformations_over_time = train(transformations, None, y, splitter)
    pred = backtest(transformations_over_time, None, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.01).all()


def test_neuralforecast_nhits() -> None:
    X, y = generate_monotonous_data(length=500)

    step = 100
    input_size = 100
    splitter = SingleWindowSplitter(train_window=400)
    model = NHITS(
        input_size=input_size,
        h=step,
        max_steps=50,
    )
    transformations = WrapNeuralForecast.from_model(model)
    transformations_over_time = train(transformations, None, y, splitter)
    pred = backtest(transformations_over_time, None, y, splitter)

    data = pd.DataFrame(
        {"ds": X[:400].index, "y": y[:400].values, "unique_id": 1.0},
    )
    nf = NeuralForecast(models=[model], freq="m")
    nf.fit(data)
    nf_pred = nf.predict()

    assert np.isclose(nf_pred["NHITS"].values, pred.squeeze().values, atol=0.01).all()
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.01).all()


def test_neuralforecast_rnn() -> None:
    X, y = generate_monotonous_data()

    step = 100
    input_size = 100
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=step)

    transformations = WrapNeuralForecast.from_model(
        RNN(
            input_size=input_size,
            h=step,
            max_steps=50,
        ),
    )
    transformations_over_time = train(transformations, None, y, splitter)
    pred = backtest(transformations_over_time, None, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.01).all()
