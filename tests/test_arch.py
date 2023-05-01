import numpy as np
from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data

from fold_wrappers.arch import WrapArch


def test_arch_univariate() -> None:
    # run_pipeline_and_check_if_results_close_univariate(
    #     model=WrapArch(init_args=dict(vol="Garch", p=1, o=0, q=1, dist="Normal")),
    #     splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    # )

    _, y = generate_sine_wave_data(length=200)
    y = np.log(y + 2.0).diff().dropna() * 100
    model = WrapArch(init_args=dict(vol="Garch", p=1, o=0, q=1, dist="Normal"))
    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.05)
    pred, _ = train_backtest(model, None, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()


# def test_arch_univariate_online() -> None:
#     run_pipeline_and_check_if_results_close_univariate(
#         model=WrapArch(init_args={"order": (1, 1, 0)}, online_mode=True),
#         splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
#     )
