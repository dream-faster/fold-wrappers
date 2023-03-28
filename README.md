<p align="center">
  <a href="https://github.com/dream-faster/fold-models/actions/workflows/test-statsforecast.yaml"><img alt="Statsforecast Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-statsforecast.yaml/badge.svg"/></a>
  <a href="https://github.com/dream-faster/fold-models/actions/workflows/test-statsmodels.yaml"><img alt="StatsModels Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-statsmodels.yaml/badge.svg"/></a>
  <a href="https://github.com/dream-faster/fold-models/actions/workflows/test-xgboost.yaml"><img alt="XGBoost Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-xgboost.yaml/badge.svg"/></a>
  <a href="https://github.com/dream-faster/fold-models/actions/workflows/test-sktime.yaml"><img alt="Sktime Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-sktime.yaml/badge.svg"/></a>
  <a href="https://github.com/dream-faster/fold-models/actions/workflows/test-prophet.yaml"><img alt="Prophet Test" src="https://github.com/dream-faster/fold-models/actions/workflows/test-prophet.yaml/badge.svg"/></a>
  <a href="https://github.com/dream-faster/fold-models/actions/workflows/test-baselines.yaml"><img alt="Baselines Tests" src="https://github.com/dream-faster/fold-models/actions/workflows/test-baselines.yaml/badge.svg"/></a>
  <a href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
</p>

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://dream-faster.github.io/fold/">
    <img src="https://raw.githubusercontent.com/dream-faster/fold-models/main/docs/images/logo.svg" alt="Logo" width="90" >
  </a>
<h3 align="center"><b>FOLD-MODELS</b><br> <i>(/fold models/)</i></h3>
  <p align="center">
    <b>Baseline models and wrappers for 3rd party libraries.
    <br/>To be used with  <a href='https://github.com/dream-faster/fold'>Fold.</a> </b><br>
    <br/>
    <a href="https://dream-faster.github.io/fold-models/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

# Available models

|                                                                                                                                                                                                                                             | Name                                   |                          Link                          | Supports<br />Online <br />updating | Wrapper Name<br />& Import Location                                            |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------- | :----------------------------------------------------: | :---------------------------------: | ------------------------------------------------------------------------------ |
| <img alt='Statsforecast Logo' src='https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png' height=64>                                                                                                      | StatsForecast                          |   [GitHub](https://github.com/Nixtla/statsforecast)    |                 ❌                  | **WrapStatsForecast**<br />`from fold_models.prophet import WrapStatsForecast` |
| <img alt='XGBoost Logo' src='https://camo.githubusercontent.com/0ea6e7814dd771f740509bbb668d251d485a6e21f12e287be7cc2275e0eab1d1/68747470733a2f2f7867626f6f73742e61692f696d616765732f6c6f676f2f7867626f6f73742d6c6f676f2e737667' height=64> | XGBoost                                |       [GitHub](https://github.com/dmlc/xgboost)        |                                     | <br />**WrapXGB**<br />`from fold_models.xgboost import WrapXGB`               |
| <img alt='Sktime Logo' src='https://github.com/sktime/sktime/raw/main/docs/source/images/sktime-logo.jpg?raw=true' height=64>                                                                                                               | Sktime                                 |       [GitHub](https://github.com/sktime/sktime)       |                 ✅                  | **WrapSktime**<br />`from fold_models.sktime import WrapSktime`                |
| <img alt='Statsmodels Logo' src='https://github.com/statsmodels/statsmodels/raw/main/docs/source/images/statsmodels-logo-v2-horizontal.svg' width=160>                                                                                      | Statsmodels                            |  [GitHub](https://github.com/statsmodels/statsmodels)  |                 ✅                  | **WrapStatsModels**<br />`from fold_models.statsmodels import WrapStatsModels` |
| <img alt='Prophet Logo' src='https://miro.medium.com/v2/resize:fit:964/0*tVCene42rgUTNv9Q.png' width=160>                                                                                                                                   | Prophet                                |     [GitHub](https://github.com/facebook/prophet)      |                 ✅                  | **WrapProphet**<br />`from fold_models.prophet import WrapProphet`             |
| <img alt='Scikit-Learn Logo' src='https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png' width=160>                                                                                              | Sklearn (natively available in `fold`) | [GitHub](https://github.com/scikit-learn/scikit-learn) |             🟡``(some)              | Sklearn doesn't need to be wrapped,<br />just pass in the models.              |

# Quickstart

You can quickly train your chosen models and get predictions by running:

```python
import pandas as pd
from fold import train_evaluate, ExpandingWindowSplitter
from fold.transformations import OnlyPredictions
from fold.models.dummy import DummyRegressor
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la",
    target_col="temperature",
    shorten=1000
)

transformations = [
    DummyRegressor(0),
    OnlyPredictions(),
]
splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
scorecard, prediction, trained_transformations = train_evaluate(
    transformations, X, y, splitter
)
```
