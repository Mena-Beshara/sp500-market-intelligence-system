# 06 — Deep learning comparison

## 1. Setup


```python
import time
import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

print(f'TensorFlow {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')
```

    TensorFlow 2.21.0
    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    

    GPU available: False
    


```python
# ── Walk-forward configuration (inherited from Notebook 05) ──
TEST_SIZE   = 252   # trading days in the test window
REFIT_EVERY = 21    # days between model retraining
LOOKBACK    = 21    # sequence length (one trading month)

# ── LSTM configuration ──
LSTM_UNITS  = 32
DROPOUT     = 0.2
EPOCHS      = 100
BATCH_SIZE  = 32
PATIENCE    = 10

# ── MLP configuration ──
MLP_HIDDEN  = 64    # single hidden layer width

# ── Seeds ──
# The walk-forward is repeated once per seed. Two runs of an earlier version
# of this notebook, identical in code, seed and data, produced LSTM RMSE
# 0.005207 and 0.006840. A single run cannot express that, so the evaluation
# reports the spread across seeds instead of one point estimate.
#
# SEED_LIST[0] is the primary seed: its forecasts feed every diagnostic
# section below, so those sections remain directly comparable to Notebook 05.
# The remaining seeds exist to measure variability.
#
# Cost scales linearly. Three seeds is roughly an hour on CPU. Adding seeds
# tightens the estimate of the spread; removing them widens the uncertainty
# about the uncertainty.
SEED_LIST = [42, 43, 44]
SEED = SEED_LIST[0]

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Inference configuration ──
N_BOOT      = 2000  # bootstrap replications for RMSE confidence intervals
BOOT_BLOCK  = 21    # moving block length (preserves error autocorrelation)
LB_LAGS     = [10, 21]  # Ljung-Box lags for residual autocorrelation

# ── Materiality threshold ──
# A bare inequality treats a 0.02% RMSE gap and a 30% gap as the same
# statement. Differences below this threshold are reported as ties.
RMSE_MATERIAL = 0.01

print(f'Configuration locked. Seeds: {SEED_LIST}')
```

    Configuration locked. Seeds: [42, 43, 44]
    

## 2. Data loading


```python
df = pd.read_parquet('../data/sp500_features.parquet')

print(f'Rows:    {len(df):,}')
print(f'Columns: {df.shape[1]}')
print(f'Range:   {df.index.min().date()} to {df.index.max().date()}')
df.head(3)
```

    Rows:    6,441
    Columns: 66
    Range:   2001-02-13 to 2026-09-25
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>log_returns</th>
      <th>return_lag_1</th>
      <th>return_lag_2</th>
      <th>return_lag_3</th>
      <th>return_lag_5</th>
      <th>...</th>
      <th>month_8</th>
      <th>month_9</th>
      <th>month_10</th>
      <th>month_11</th>
      <th>month_12</th>
      <th>vol_above_avg</th>
      <th>vol_rank_30</th>
      <th>high_vol_regime</th>
      <th>vol_expansion</th>
      <th>vol_ratio_10_60</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-02-13</th>
      <td>1330.310059</td>
      <td>1336.619995</td>
      <td>1317.510010</td>
      <td>1318.800049</td>
      <td>1075200000</td>
      <td>-0.008690</td>
      <td>0.011758</td>
      <td>-0.013425</td>
      <td>-0.006254</td>
      <td>-0.001515</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.440476</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.614745</td>
    </tr>
    <tr>
      <th>2001-02-14</th>
      <td>1318.800049</td>
      <td>1320.729980</td>
      <td>1304.719971</td>
      <td>1315.920044</td>
      <td>1150300000</td>
      <td>-0.002186</td>
      <td>-0.008690</td>
      <td>0.011758</td>
      <td>-0.013425</td>
      <td>-0.008444</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.329365</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.618165</td>
    </tr>
    <tr>
      <th>2001-02-15</th>
      <td>1315.920044</td>
      <td>1331.290039</td>
      <td>1315.920044</td>
      <td>1326.609985</td>
      <td>1153700000</td>
      <td>0.008091</td>
      <td>-0.002186</td>
      <td>-0.008690</td>
      <td>0.011758</td>
      <td>-0.006254</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.194444</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.639375</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 66 columns</p>
</div>




```python
# Every value below comes from Notebook 05's export, read without a default:
# a missing file or key stops the notebook instead of printing a stale figure.
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text())

nb05 = metrics['notebook_05']

GARCH_RMSE  = nb05['wf_rmse_garch']
GARCH_MAE   = nb05['wf_mae_garch']
PERS_RMSE   = nb05['wf_rmse_persistence']
PERS_MAE    = nb05['wf_mae_persistence']
GARCH_LABEL = nb05['best_label']
GARCH_PERSISTENCE = nb05['garch_persistence']

garch_vs_pers = (PERS_RMSE - GARCH_RMSE) / PERS_RMSE * 100

display(Markdown(f"""
### Notebook 05 baselines

| Model | RMSE | MAE |
|---|---|---|
| Persistence | {PERS_RMSE:.6f} | {PERS_MAE:.6f} |
| {GARCH_LABEL} | {GARCH_RMSE:.6f} | {GARCH_MAE:.6f} |

GARCH beat persistence by {garch_vs_pers:.1f}% RMSE.
"""))
```



### Notebook 05 baselines

| Model | RMSE | MAE |
|---|---|---|
| Persistence | 0.007562 | 0.005686 |
| GJR-GARCH(1,1,1) — skewed Student's t | 0.005236 | 0.004005 |

GARCH beat persistence by 30.8% RMSE.




```python
# Estimated parameters of the selected model, excluding the constant mean.
# Notebook 05 counts them on the fitted model and exports the total, so the
# figure stays correct whichever specification wins the AIC comparison.
garch_n_params = nb05['garch_n_params']

# What each parameter encodes: beta, omega, alpha and gamma in the GJR variance
# equation, then one entry per shape parameter of the innovation distribution
# Notebook 05 selected. The check stops the notebook if the list and the count
# ever disagree.
shape_mechanisms = {'nu': 'tail thickness', 'eta': 'tail thickness',
                    'lambda': 'skew'}
mechanisms = ['persistence', 'mean reversion', 'shock sensitivity',
              'leverage asymmetry']
mechanisms += [shape_mechanisms[name] for name in nb05['garch_shape']]
assert len(mechanisms) == garch_n_params, (
    f'{len(mechanisms)} mechanisms listed for {garch_n_params} parameters; '
    'update this list for the model Notebook 05 selected')
mechanism_text = ', '.join(mechanisms[:-1]) + ', and ' + mechanisms[-1]

display(Markdown(f"""
LSTM and MLP versus {GARCH_LABEL} on one-step-ahead volatility forecasting.

Notebook 04 found no evidence that return direction could be forecast more
accurately than the historical mean baseline. Notebook 05 showed that
volatility is forecastable: {GARCH_LABEL} beat persistence by
{garch_vs_pers:.1f}% RMSE over a {TEST_SIZE}-day walk-forward window, using
{garch_n_params} parameters that each encode a specific financial mechanism
({mechanism_text}).

This notebook tests whether neural networks, given access to engineered
features GARCH never sees, can match or beat a model built from financial
structure. Two architectures are compared: an LSTM, which processes sequences
and can learn temporal dependencies, and a feed-forward MLP, which sees the
same features flattened, without sequential structure. Comparing the two
isolates whether sequence learning adds value beyond the features themselves.

The evaluation protocol is inherited from Notebook 05 unchanged, with one
addition: the walk-forward is repeated across {len(SEED_LIST)} random seeds,
because a single run of a neural network is one draw from a distribution this
notebook cannot otherwise characterise.
"""))
```



LSTM and MLP versus GJR-GARCH(1,1,1) — skewed Student's t on one-step-ahead volatility forecasting.

Notebook 04 found no evidence that return direction could be forecast more
accurately than the historical mean baseline. Notebook 05 showed that
volatility is forecastable: GJR-GARCH(1,1,1) — skewed Student's t beat persistence by
30.8% RMSE over a 252-day walk-forward window, using
6 parameters that each encode a specific financial mechanism
(persistence, mean reversion, shock sensitivity, leverage asymmetry, tail thickness, and skew).

This notebook tests whether neural networks, given access to engineered
features GARCH never sees, can match or beat a model built from financial
structure. Two architectures are compared: an LSTM, which processes sequences
and can learn temporal dependencies, and a feed-forward MLP, which sees the
same features flattened, without sequential structure. Comparing the two
isolates whether sequence learning adds value beyond the features themselves.

The evaluation protocol is inherited from Notebook 05 unchanged, with one
addition: the walk-forward is repeated across 3 random seeds,
because a single run of a neural network is one draw from a distribution this
notebook cannot otherwise characterise.




```python
display(Markdown(f"""
## 3. Where this fits

| Notebook | Question | Result |
|---|---|---|
| 04 — Baseline forecasting | Can ARIMA predict return direction? | No. ARIMA matched the historical mean baseline. Null result consistent with weak-form efficiency. |
| 05 — Volatility forecasting | Can GARCH-family models forecast volatility? | Yes. {GARCH_LABEL} beat persistence by {garch_vs_pers:.1f}% RMSE. Selected as the production model. |
| **06 — Deep learning comparison** | **Can neural networks beat the selected model?** | **Tested below.** |

The analytical thread across these notebooks is a narrowing search. Direction
proved unforecastable. Volatility proved forecastable using models that encode
financial structure. The remaining question is whether flexible models with
richer inputs can do better, and if so, whether the gain comes from temporal
modelling (LSTM) or from the feature set alone (MLP).
"""))
```



## 3. Where this fits

| Notebook | Question | Result |
|---|---|---|
| 04 — Baseline forecasting | Can ARIMA predict return direction? | No. ARIMA matched the historical mean baseline. Null result consistent with weak-form efficiency. |
| 05 — Volatility forecasting | Can GARCH-family models forecast volatility? | Yes. GJR-GARCH(1,1,1) — skewed Student's t beat persistence by 30.8% RMSE. Selected as the production model. |
| **06 — Deep learning comparison** | **Can neural networks beat the selected model?** | **Tested below.** |

The analytical thread across these notebooks is a narrowing search. Direction
proved unforecastable. Volatility proved forecastable using models that encode
financial structure. The remaining question is whether flexible models with
richer inputs can do better, and if so, whether the gain comes from temporal
modelling (LSTM) or from the feature set alone (MLP).



## 4. Research hypothesis

GARCH models encode volatility dynamics through a small number of
economically interpretable parameters. Deep neural networks provide
substantially greater flexibility and may exploit nonlinear interactions
between engineered features that a parametric model cannot represent.

The hypothesis tested here is whether this additional flexibility translates
into superior out-of-sample volatility forecasts under an identical
walk-forward evaluation protocol.

Two sub-hypotheses follow from it. First, if flexibility helps, at least one
neural network should beat GJR-GARCH on RMSE. Second, if sequential structure
is what the flexibility buys, the LSTM should beat the MLP. The two questions
are separable, and the answers point to different next steps: a win for neural
networks over GARCH argues for capacity, while a win for the LSTM over the MLP
argues specifically for temporal modelling.

The null result is informative in either direction. If neither network wins,
the conclusion is that structural knowledge outperforms capacity on this data
volume, which is a finding about the problem rather than a failure of the
method.

A third question emerged from running this notebook rather than from planning
it: whether a neural network result on this data is stable enough to report as
a single number at all. Section 10 addresses that directly.

## 5. Evaluation protocol

Inherited from Notebook 05 without modification. The target is the absolute
daily log return, a realised volatility proxy. The test window covers the
final 252 trading days. Models are retrained every 21 days on an expanding
window. The benchmark is persistence: yesterday's absolute return forecasts
tomorrow's. RMSE and MAE are computed over the full test window.

### Why MSE trains the networks but RMSE ranks them

The neural networks are trained using mean squared error because it provides
smooth gradients and places greater emphasis on larger forecasting errors,
which matters for a risk application where missing a volatility spike is
costlier than a small error on a quiet day. Performance is reported using both
RMSE and MAE to allow comparison with Notebook 05. RMSE is the primary ranking
metric because it penalises large volatility misses more heavily and is
directly comparable to the GARCH results. MAE provides an interpretable
measure of average forecast error in the units of the target.

### What each model predicts

GARCH models the conditional variance process that generates returns. Its
output is a distributional parameter (sigma) from which expected absolute
returns are derived. The neural networks predict the realised volatility proxy
directly as a point estimate, with no distributional assumptions. GARCH says
"the variance of the return distribution is X." The LSTM and MLP say
"tomorrow's absolute return will be Y." The evaluation compresses all three
into the same metric space for comparison, which is standard but worth noting.

### Reproducibility: what the seed does and does not control

Setting a seed fixes weight initialisation, dropout masks and data shuffling.
It does not make TensorFlow deterministic on CPU. Operations such as matrix
multiplication and gradient reduction are parallelised across threads, and
floating-point addition is not associative, so the order in which threads
finish changes the sum in the final bits. Those differences compound through
backpropagation, and early stopping converts them into discrete outcomes: a
validation loss that crosses its threshold one epoch earlier produces a
different set of weights.

An earlier version of this notebook stated that model ranking was stable and
only the final decimal places moved. Two executions of that version, identical
in code, seed, data and hardware, refuted it. LSTM walk-forward RMSE came out
at 0.005207 and then 0.006840, a 31% degradation. Residual autocorrelation at
21 lags moved from p = 0.94 to p = 3.6e-34. The calibration slope fell from
0.645 to 0.079. Those are not decimal places; they are different conclusions
about the same model.

This notebook therefore repeats the walk-forward across several seeds and
reports the spread. GARCH, by contrast, is deterministic: the same data
produces the same coefficients every time, which is itself a deployment
consideration and not only a statistical one.

## 6. Target and feature selection


```python
TARGET_COL = 'log_returns'

FEATURE_COLS = [
    'log_returns',     # the return series itself
    'vol_roll_10',     # short-term rolling volatility
    'vol_roll_21',     # medium-term rolling volatility
    'vol_roll_60',     # longer-term volatility context
    'rsi_14',          # momentum indicator
    'atr_pct_14',      # average true range as a share of the close
    'bb_width',        # Bollinger Band width
    'vol_rank_30',     # volatility percentile rank
    'vol_ratio_10_60', # short vs medium volatility ratio
    'volume_rel_21',   # yesterday's volume relative to its 21-day average]
]

missing = [c for c in FEATURE_COLS if c not in df.columns]
assert not missing, f'Missing columns: {missing}'

n_features = len(FEATURE_COLS)
print(f'Features: {n_features}')
print(f'Target:   |{TARGET_COL}|')
```

    Features: 10
    Target:   |log_returns|
    

### Feature rationale

GARCH uses only the return series and its own conditional variance recursion.
The neural networks get a richer input set: three rolling volatility windows
(10, 21, 60 days), RSI, ATR as a share of the closing price, Bollinger Band
width, a volatility percentile rank, a short-to-medium volatility ratio, and
yesterday's volume relative to its trailing 21-day average. ATR and volume
enter in these scale-free forms because, measured in index points and shares,
both drift with the market across the sample.

Lagged return features from Notebook 03 are excluded because the 21-day
lookback window already provides that information through the `log_returns`
channel. Calendar dummies are excluded: SARIMA found no significant seasonal
structure in Notebook 04.

## 7. Sequence construction


```python
def create_sequences(X, y, lookback):
    """Build (sequence, target) pairs for supervised learning.

    sequence[i] = features[i : i + lookback]
    target[i]   = |return[i + lookback]|

    At prediction time, all features in the sequence are known up to and
    including the current close, and the target is tomorrow's absolute return.
    """
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


X_all = df[FEATURE_COLS].values
y_all = df[TARGET_COL].abs().values

X_seq, y_seq = create_sequences(X_all, y_all, LOOKBACK)

test_start = len(X_seq) - TEST_SIZE
pred_dates = df.index[LOOKBACK + test_start : LOOKBACK + test_start + TEST_SIZE]

print(f'Total sequences: {len(X_seq):,}')
print(f'Training pool:   {test_start:,}')
print(f'Test sequences:  {TEST_SIZE}')
print(f'Test window:     {pred_dates[0].date()} to {pred_dates[-1].date()}')
print(f'Sequence shape:  {X_seq[0].shape}')
```

    Total sequences: 6,420
    Training pool:   6,168
    Test sequences:  252
    Test window:     2025-09-25 to 2026-09-25
    Sequence shape:  (21, 10)
    

The 21-day lookback matches the rolling volatility window used throughout the
project. Longer windows (60 or 120 days) increase the parameter count without
clear evidence of long-range dependencies the rolling features do not already
capture.

## 8. Architectures

```
LSTM
21 days x 10 features --> LSTM (32 units) --> Dropout --> Dense (1, softplus)
  ordering preserved        recurrent state

MLP
21 days x 10 features --> Flatten (210) --> Dense (64, relu) --> Dropout --> Dense (1, softplus)
  ordering discarded        one long vector
```

The output activation is `softplus` rather than `relu`. Both enforce
non-negativity, since the target is an absolute return, but `relu` has a
zero-gradient region that can stall learning when predictions are near zero.
`softplus`, log(1 + exp(x)), is smooth everywhere and allows gradients to flow
at all output levels. The MLP's hidden layer uses `relu`: the zero-gradient
concern applies to the output layer, where predictions cluster near zero, not
to a hidden layer operating across a wider activation range.

The MLP exists as a control. It receives the same 21 x 10 = 210 input values
flattened into a single vector, discarding the sequential structure. If the MLP
matches the LSTM, temporal modelling adds nothing and the feature set alone
carries whatever signal exists. If the LSTM materially outperforms it, there is
evidence of exploitable sequential structure beyond what the rolling features
already summarise.


```python
def build_lstm(lookback, n_features, units=LSTM_UNITS):
    """Single-layer LSTM for one-step-ahead volatility forecasting."""
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(units, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1, activation='softplus'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_mlp(lookback, n_features, hidden=MLP_HIDDEN):
    """Feed-forward baseline: same inputs, no sequential structure."""
    model = Sequential([
        Input(shape=(lookback, n_features)),
        Flatten(),
        Dense(hidden, activation='relu'),
        Dropout(DROPOUT),
        Dense(1, activation='softplus'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


_demo_lstm = build_lstm(LOOKBACK, n_features)
lstm_params = int(sum(np.prod(w.shape) for w in _demo_lstm.trainable_weights))
del _demo_lstm

_demo_mlp = build_mlp(LOOKBACK, n_features)
mlp_params = int(sum(np.prod(w.shape) for w in _demo_mlp.trainable_weights))
del _demo_mlp

print(f'LSTM trainable parameters: {lstm_params:,}')
print(f'MLP trainable parameters:  {mlp_params:,}')
print(f'MLP / LSTM ratio:          {mlp_params / lstm_params:.1f}x')
print(f'GARCH parameters:          {garch_n_params}')
```

    LSTM trainable parameters: 5,537
    MLP trainable parameters:  13,569
    MLP / LSTM ratio:          2.5x
    GARCH parameters:          6
    

### A note on feature scaling

Standardisation is performed separately within each walk-forward training
window using statistics computed only from the training data. The fitted
scaler is then applied unchanged to the corresponding test block. This
prevents information leakage while maintaining comparable feature scales
during optimisation.

Scaling does not destroy temporal information. It is a per-feature affine
transformation applied identically to every timestep, so the ordering, the
relative movements, and the autocorrelation structure within each sequence are
all preserved. What changes is the numerical range the optimiser works in,
which matters because gradient descent converges poorly when input features
differ by orders of magnitude, as they do here, with log returns near 0.01.

## 9. Walk-forward evaluation

The walk-forward runs once per seed. Everything except the random seed is held
constant: same data, same splits, same refit schedule, same architecture.


```python
def run_walk_forward(seed, verbose=True):
    """One complete walk-forward for both architectures at a given seed.

    Returns a dict with forecasts, timings, epoch counts and training
    histories. Everything except the seed is identical across calls, so
    differences between returned results are attributable to the seed and to
    the non-determinism the seed does not control.
    """
    n_blocks = (TEST_SIZE + REFIT_EVERY - 1) // REFIT_EVERY
    lstm_fc = np.zeros(TEST_SIZE)
    mlp_fc = np.zeros(TEST_SIZE)
    hist_lstm, hist_mlp = [], []
    times_lstm, times_mlp = [], []
    epochs_lstm, epochs_mlp = [], []

    for block in range(n_blocks):
        block_start = block * REFIT_EVERY
        block_end = min(block_start + REFIT_EVERY, TEST_SIZE)

        train_end = test_start + block_start
        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train.reshape(-1, n_features)
        ).reshape(X_train.shape)

        val_size = max(int(0.1 * len(X_train_scaled)), LOOKBACK)
        X_val, y_val = X_train_scaled[-val_size:], y_train[-val_size:]
        X_fit, y_fit = X_train_scaled[:-val_size], y_train[:-val_size]

        X_test_block = X_seq[test_start + block_start : test_start + block_end]
        X_test_scaled = scaler.transform(
            X_test_block.reshape(-1, n_features)
        ).reshape(X_test_block.shape)

        es = EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                           monitor='val_loss')

        # ── LSTM ──
        tf.random.set_seed(seed + block)
        lstm_model = build_lstm(LOOKBACK, n_features)
        t0 = time.perf_counter()
        h = lstm_model.fit(X_fit, y_fit, validation_data=(X_val, y_val),
                           epochs=EPOCHS, batch_size=BATCH_SIZE,
                           callbacks=[es], verbose=0)
        times_lstm.append(time.perf_counter() - t0)
        hist_lstm.append(h.history)
        epochs_lstm.append(len(h.history['loss']))
        lstm_fc[block_start:block_end] = lstm_model.predict(
            X_test_scaled, verbose=0).flatten()

        # ── MLP ──
        tf.random.set_seed(seed + block)
        mlp_model = build_mlp(LOOKBACK, n_features)
        t0 = time.perf_counter()
        h = mlp_model.fit(X_fit, y_fit, validation_data=(X_val, y_val),
                          epochs=EPOCHS, batch_size=BATCH_SIZE,
                          callbacks=[es], verbose=0)
        times_mlp.append(time.perf_counter() - t0)
        hist_mlp.append(h.history)
        epochs_mlp.append(len(h.history['loss']))
        mlp_fc[block_start:block_end] = mlp_model.predict(
            X_test_scaled, verbose=0).flatten()

        if verbose:
            print(f'  block {block+1:2d}/{n_blocks}: '
                  f'LSTM {epochs_lstm[-1]:3d}ep {times_lstm[-1]:5.1f}s | '
                  f'MLP {epochs_mlp[-1]:3d}ep {times_mlp[-1]:5.1f}s')

    return {
        'seed': seed,
        'n_blocks': n_blocks,
        'lstm_forecasts': lstm_fc,
        'mlp_forecasts': mlp_fc,
        'hist_lstm': hist_lstm,
        'hist_mlp': hist_mlp,
        'epochs_lstm': epochs_lstm,
        'epochs_mlp': epochs_mlp,
        'time_lstm': sum(times_lstm),
        'time_mlp': sum(times_mlp),
        'last_lstm': lstm_model,
        'last_mlp': mlp_model,
        'last_scaler': scaler,
    }


print(f'Walk-forward across {len(SEED_LIST)} seeds. This is the slow cell.')
```

    Walk-forward across 3 seeds. This is the slow cell.
    


```python
actual = y_seq[test_start : test_start + TEST_SIZE]
persistence = y_seq[test_start - 1 : test_start + TEST_SIZE - 1]

pers_rmse_check = root_mean_squared_error(actual, persistence)
pers_mae_check = mean_absolute_error(actual, persistence)

print(f'Persistence RMSE (this window): {pers_rmse_check:.6f}')
print(f'NB05 persistence RMSE:          {PERS_RMSE:.6f}')

if abs(pers_rmse_check - PERS_RMSE) / PERS_RMSE > 0.05:
    print('\nWARNING: persistence RMSE differs by >5% from NB05. '
          'Check test window alignment.')
```

    Persistence RMSE (this window): 0.007562
    NB05 persistence RMSE:          0.007562
    


```python
runs = {}
wf_start_all = time.perf_counter()

for seed in SEED_LIST:
    print(f'Seed {seed}:')
    runs[seed] = run_walk_forward(seed)
    r = runs[seed]
    r['lstm_rmse'] = root_mean_squared_error(actual, r['lstm_forecasts'])
    r['lstm_mae'] = mean_absolute_error(actual, r['lstm_forecasts'])
    r['mlp_rmse'] = root_mean_squared_error(actual, r['mlp_forecasts'])
    r['mlp_mae'] = mean_absolute_error(actual, r['mlp_forecasts'])
    print(f'  -> LSTM RMSE {r["lstm_rmse"]:.6f} | '
          f'MLP RMSE {r["mlp_rmse"]:.6f} | '
          f'{r["time_lstm"] + r["time_mlp"]:.0f}s\n')

wf_total = time.perf_counter() - wf_start_all

# The primary seed drives every diagnostic section below.
primary = runs[SEED]
lstm_forecasts = primary['lstm_forecasts']
mlp_forecasts = primary['mlp_forecasts']
lstm_rmse = primary['lstm_rmse']
lstm_mae = primary['lstm_mae']
mlp_rmse = primary['mlp_rmse']
mlp_mae = primary['mlp_mae']
n_blocks = primary['n_blocks']
wf_total_lstm = primary['time_lstm']
wf_total_mlp = primary['time_mlp']
epochs_used_lstm = primary['epochs_lstm']
epochs_used_mlp = primary['epochs_mlp']
training_histories_lstm = primary['hist_lstm']
training_histories_mlp = primary['hist_mlp']
last_lstm = primary['last_lstm']
last_mlp = primary['last_mlp']
last_scaler = primary['last_scaler']

lstm_errors = actual - lstm_forecasts
mlp_errors = actual - mlp_forecasts
pers_errors = actual - persistence

print(f'All seeds complete in {wf_total:.0f}s')
```

    Seed 42:
    

      block  1/12: LSTM  16ep  25.9s | MLP  34ep  17.5s
    

      block  2/12: LSTM  62ep  80.9s | MLP  55ep  26.9s
    

    WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001BF86819A80> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001BF879EE340> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

      block  3/12: LSTM  57ep  75.1s | MLP  71ep  37.8s
    

      block  4/12: LSTM  14ep  20.1s | MLP  28ep  15.7s
    

      block  5/12: LSTM  15ep  22.6s | MLP  35ep  18.1s
    

      block  6/12: LSTM  77ep 102.7s | MLP  44ep  26.4s
    

      block  7/12: LSTM  88ep 117.8s | MLP  40ep  22.1s
    

      block  8/12: LSTM  16ep  22.6s | MLP  36ep  19.0s
    

      block  9/12: LSTM  73ep  99.0s | MLP  32ep  18.0s
    

      block 10/12: LSTM  58ep  76.8s | MLP  18ep   9.6s
    

      block 11/12: LSTM  14ep  20.1s | MLP  26ep  13.9s
    

      block 12/12: LSTM  16ep  20.2s | MLP  23ep  11.5s
      -> LSTM RMSE 0.005387 | MLP RMSE 0.007588 | 921s
    
    Seed 43:
    

      block  1/12: LSTM  15ep  21.0s | MLP  25ep  13.4s
    

      block  2/12: LSTM  15ep  21.2s | MLP  47ep  24.9s
    

      block  3/12: LSTM  63ep  82.6s | MLP  37ep  18.5s
    

      block  4/12: LSTM  56ep  73.7s | MLP  53ep  27.5s
    

      block  5/12: LSTM  15ep  21.5s | MLP  69ep  34.2s
    

      block  6/12: LSTM  16ep  22.4s | MLP  45ep  22.3s
    

      block  7/12: LSTM  15ep  20.5s | MLP  42ep  20.0s
    

      block  8/12: LSTM  81ep 105.0s | MLP  40ep  19.6s
    

      block  9/12: LSTM  15ep  21.1s | MLP  26ep  14.0s
    

      block 10/12: LSTM  57ep  74.4s | MLP  22ep  11.7s
    

      block 11/12: LSTM  14ep  20.6s | MLP  38ep  19.7s
    

      block 12/12: LSTM  15ep  19.2s | MLP  38ep  19.2s
      -> LSTM RMSE 0.005473 | MLP RMSE 0.008223 | 748s
    
    Seed 44:
    

      block  1/12: LSTM  54ep  70.7s | MLP  25ep  12.7s
    

      block  2/12: LSTM  15ep  22.1s | MLP  40ep  20.7s
    

      block  3/12: LSTM  17ep  23.9s | MLP  62ep  30.5s
    

      block  4/12: LSTM  15ep  21.6s | MLP  32ep  15.1s
    

      block  5/12: LSTM  79ep 103.4s | MLP  46ep  21.4s
    

      block  6/12: LSTM  69ep  89.7s | MLP  28ep  13.3s
    

      block  7/12: LSTM  75ep  99.0s | MLP  23ep  12.1s
    

      block  8/12: LSTM  78ep 103.6s | MLP  26ep  13.5s
    

      block  9/12: LSTM  16ep  22.9s | MLP  35ep  19.2s
    

      block 10/12: LSTM  62ep  84.1s | MLP  34ep  18.7s
    

      block 11/12: LSTM  16ep  23.5s | MLP  24ep  12.7s
    

      block 12/12: LSTM  15ep  19.0s | MLP  29ep  14.6s
      -> LSTM RMSE 0.005417 | MLP RMSE 0.007547 | 888s
    
    All seeds complete in 2575s
    

## 10. Seed stability

Before comparing the networks against GARCH, the question is whether a single
network run is a meaningful quantity to compare at all.


```python
seed_rows = []
for seed in SEED_LIST:
    r = runs[seed]
    seed_rows.append({
        'Seed': seed,
        'LSTM RMSE': r['lstm_rmse'],
        'LSTM MAE': r['lstm_mae'],
        'MLP RMSE': r['mlp_rmse'],
        'LSTM epochs (mean)': np.mean(r['epochs_lstm']),
    })

seed_df = pd.DataFrame(seed_rows)
display(seed_df.style.format({
    'LSTM RMSE': '{:.6f}', 'LSTM MAE': '{:.6f}',
    'MLP RMSE': '{:.6f}', 'LSTM epochs (mean)': '{:.1f}',
}).hide(axis='index'))

lstm_rmses = seed_df['LSTM RMSE'].values
mlp_rmses = seed_df['MLP RMSE'].values

lstm_rmse_mean = lstm_rmses.mean()
lstm_rmse_sd = lstm_rmses.std(ddof=1) if len(lstm_rmses) > 1 else 0.0
lstm_rmse_min, lstm_rmse_max = lstm_rmses.min(), lstm_rmses.max()
lstm_spread_pct = (lstm_rmse_max - lstm_rmse_min) / lstm_rmse_min * 100

mlp_rmse_mean = mlp_rmses.mean()
mlp_rmse_sd = mlp_rmses.std(ddof=1) if len(mlp_rmses) > 1 else 0.0
mlp_spread_pct = (mlp_rmses.max() - mlp_rmses.min()) / mlp_rmses.min() * 100

# Does the seed change the answer to the notebook's question?
seeds_beating_garch = int((lstm_rmses < GARCH_RMSE).sum())
seeds_beating_pers = int((lstm_rmses < pers_rmse_check).sum())

print(f'LSTM RMSE across {len(SEED_LIST)} seeds:')
print(f'  mean {lstm_rmse_mean:.6f}, sd {lstm_rmse_sd:.6f}')
print(f'  range {lstm_rmse_min:.6f} to {lstm_rmse_max:.6f} '
      f'({lstm_spread_pct:.1f}% spread)')
print(f'  seeds beating GARCH ({GARCH_RMSE:.6f}): '
      f'{seeds_beating_garch}/{len(SEED_LIST)}')
print(f'  seeds beating persistence ({pers_rmse_check:.6f}): '
      f'{seeds_beating_pers}/{len(SEED_LIST)}')
```


<style type="text/css">
</style>
<table id="T_641c0">
  <thead>
    <tr>
      <th id="T_641c0_level0_col0" class="col_heading level0 col0" >Seed</th>
      <th id="T_641c0_level0_col1" class="col_heading level0 col1" >LSTM RMSE</th>
      <th id="T_641c0_level0_col2" class="col_heading level0 col2" >LSTM MAE</th>
      <th id="T_641c0_level0_col3" class="col_heading level0 col3" >MLP RMSE</th>
      <th id="T_641c0_level0_col4" class="col_heading level0 col4" >LSTM epochs (mean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_641c0_row0_col0" class="data row0 col0" >42</td>
      <td id="T_641c0_row0_col1" class="data row0 col1" >0.005387</td>
      <td id="T_641c0_row0_col2" class="data row0 col2" >0.003935</td>
      <td id="T_641c0_row0_col3" class="data row0 col3" >0.007588</td>
      <td id="T_641c0_row0_col4" class="data row0 col4" >42.2</td>
    </tr>
    <tr>
      <td id="T_641c0_row1_col0" class="data row1 col0" >43</td>
      <td id="T_641c0_row1_col1" class="data row1 col1" >0.005473</td>
      <td id="T_641c0_row1_col2" class="data row1 col2" >0.003920</td>
      <td id="T_641c0_row1_col3" class="data row1 col3" >0.008223</td>
      <td id="T_641c0_row1_col4" class="data row1 col4" >31.4</td>
    </tr>
    <tr>
      <td id="T_641c0_row2_col0" class="data row2 col0" >44</td>
      <td id="T_641c0_row2_col1" class="data row2 col1" >0.005417</td>
      <td id="T_641c0_row2_col2" class="data row2 col2" >0.003885</td>
      <td id="T_641c0_row2_col3" class="data row2 col3" >0.007547</td>
      <td id="T_641c0_row2_col4" class="data row2 col4" >42.6</td>
    </tr>
  </tbody>
</table>



    LSTM RMSE across 3 seeds:
      mean 0.005426, sd 0.000043
      range 0.005387 to 0.005473 (1.6% spread)
      seeds beating GARCH (0.005236): 0/3
      seeds beating persistence (0.007562): 3/3
    


```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[str(s) for s in SEED_LIST], y=lstm_rmses,
    mode='markers', name=f'LSTM ({LSTM_UNITS} units)',
    marker=dict(color='#00d4aa', size=13),
))
fig.add_trace(go.Scatter(
    x=[str(s) for s in SEED_LIST], y=mlp_rmses,
    mode='markers', name=f'MLP ({MLP_HIDDEN} hidden)',
    marker=dict(color='#ffd700', size=13),
))
fig.add_hline(y=GARCH_RMSE, line_dash='dash', line_color='#00aaff',
              annotation_text=f'GARCH {GARCH_RMSE:.6f}')
fig.add_hline(y=pers_rmse_check, line_dash='dot', line_color='#ff6b6b',
              annotation_text=f'Persistence {pers_rmse_check:.6f}')
fig.update_layout(
    template='plotly_dark',
    title='Walk-forward RMSE by random seed',
    xaxis_title='Seed', yaxis_title='RMSE',
    height=420,
)
fig.show()

if seeds_beating_garch == 0:
    stability_verdict = (
        f"No seed produced an LSTM that beat {GARCH_LABEL}. The conclusion "
        "does not depend on which run is reported."
    )
elif seeds_beating_garch == len(SEED_LIST):
    stability_verdict = (
        f"Every seed produced an LSTM that beat {GARCH_LABEL}. The conclusion "
        "does not depend on which run is reported."
    )
else:
    stability_verdict = (
        f"**{seeds_beating_garch} of {len(SEED_LIST)} seeds** produced an LSTM "
        f"that beat {GARCH_LABEL} and "
        f"{len(SEED_LIST) - seeds_beating_garch} did not. The answer to this "
        "notebook's central question depends on which run happens to be "
        "reported, which means no single run should be reported as the answer."
    )

if mlp_spread_pct < lstm_spread_pct / 5:
    arch_note = (
        f"The MLP's spread is {mlp_spread_pct:.1f}% against the LSTM's "
        f"{lstm_spread_pct:.1f}%, on identical data, splits and hardware. "
        f"The instability is therefore a property of the recurrent "
        f"architecture rather than of neural networks on this problem. "
        f"Recurrence applies the same weight matrix once per timestep, so a "
        f"difference in the low-order bits is amplified {LOOKBACK} times "
        f"before the loss is computed. The feed-forward path has no such "
        f"chain."
    )
else:
    arch_note = (
        f"Both architectures show comparable spread ({lstm_spread_pct:.1f}% "
        f"LSTM, {mlp_spread_pct:.1f}% MLP), so the instability is not "
        f"specific to recurrence."
    )

display(Markdown(f"""
### What the spread means

LSTM walk-forward RMSE ranges from {lstm_rmse_min:.6f} to {lstm_rmse_max:.6f}
across {len(SEED_LIST)} seeds, a spread of {lstm_spread_pct:.1f}% of the
smaller value, with standard deviation {lstm_rmse_sd:.6f}. The MLP spread is
{mlp_spread_pct:.1f}%.

{stability_verdict}

Every element of these runs is identical except the seed: same data, same
splits, same architecture, same refit schedule, same hardware. The variation
is not a modelling choice, it is the floor of what this setup can resolve. Any
LSTM-versus-GARCH difference smaller than {lstm_spread_pct:.0f}% is inside
that floor.

GARCH has no equivalent spread. Maximum likelihood on a fixed sample returns
the same coefficients on every run, so its {GARCH_RMSE:.6f} is a property of
the data rather than of one execution. For a model whose output feeds a daily
risk report, that difference is not a technicality: a forecast that changes
when the pipeline is re-run is a forecast that has to be versioned, logged and
explained.

The sections that follow use the primary seed ({SEED}) so the diagnostics
remain directly comparable to Notebook 05. Read every figure in them as one
draw from the distribution above.
"""))
```





### What the spread means

LSTM walk-forward RMSE ranges from 0.005387 to 0.005473
across 3 seeds, a spread of 1.6% of the
smaller value, with standard deviation 0.000043. The MLP spread is
9.0%.

No seed produced an LSTM that beat GJR-GARCH(1,1,1) — skewed Student's t. The conclusion does not depend on which run is reported.

Every element of these runs is identical except the seed: same data, same
splits, same architecture, same refit schedule, same hardware. The variation
is not a modelling choice, it is the floor of what this setup can resolve. Any
LSTM-versus-GARCH difference smaller than 2% is inside
that floor.

GARCH has no equivalent spread. Maximum likelihood on a fixed sample returns
the same coefficients on every run, so its 0.005236 is a property of
the data rather than of one execution. For a model whose output feeds a daily
risk report, that difference is not a technicality: a forecast that changes
when the pipeline is re-run is a forecast that has to be versioned, logged and
explained.

The sections that follow use the primary seed (42) so the diagnostics
remain directly comparable to Notebook 05. Read every figure in them as one
draw from the distribution above.



## 11. Results


```python
lstm_vs_pers_rmse = (pers_rmse_check - lstm_rmse) / pers_rmse_check * 100
lstm_vs_garch_rmse = (GARCH_RMSE - lstm_rmse) / GARCH_RMSE * 100
mlp_vs_pers_rmse = (pers_rmse_check - mlp_rmse) / pers_rmse_check * 100
mlp_vs_garch_rmse = (GARCH_RMSE - mlp_rmse) / GARCH_RMSE * 100

comparison = pd.DataFrame({
    'RMSE': [pers_rmse_check, GARCH_RMSE, mlp_rmse, lstm_rmse],
    'MAE':  [pers_mae_check, GARCH_MAE, mlp_mae, lstm_mae],
}, index=['Persistence', f'{GARCH_LABEL} (NB05)',
          f'MLP ({MLP_HIDDEN} hidden)', f'LSTM ({LSTM_UNITS} units)'])

display(comparison)
print(f'\nPrimary seed {SEED}:')
print(f'  LSTM vs persistence: {lstm_vs_pers_rmse:+.1f}% RMSE')
print(f'  LSTM vs GARCH:       {lstm_vs_garch_rmse:+.1f}% RMSE')
print(f'  MLP vs persistence:  {mlp_vs_pers_rmse:+.1f}% RMSE')
print(f'  MLP vs GARCH:        {mlp_vs_garch_rmse:+.1f}% RMSE')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RMSE</th>
      <th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Persistence</th>
      <td>0.007562</td>
      <td>0.005686</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — skewed Student's t (NB05)</th>
      <td>0.005236</td>
      <td>0.004005</td>
    </tr>
    <tr>
      <th>MLP (64 hidden)</th>
      <td>0.007588</td>
      <td>0.005570</td>
    </tr>
    <tr>
      <th>LSTM (32 units)</th>
      <td>0.005387</td>
      <td>0.003935</td>
    </tr>
  </tbody>
</table>
</div>


    
    Primary seed 42:
      LSTM vs persistence: +28.8% RMSE
      LSTM vs GARCH:       -2.9% RMSE
      MLP vs persistence:  -0.4% RMSE
      MLP vs GARCH:        -44.9% RMSE
    


```python
lstm_garch_rel = abs(lstm_rmse - GARCH_RMSE) / GARCH_RMSE
mlp_garch_rel = abs(mlp_rmse - GARCH_RMSE) / GARCH_RMSE
lstm_garch_tie = lstm_garch_rel < RMSE_MATERIAL
mlp_garch_tie = mlp_garch_rel < RMSE_MATERIAL

garch_wins = lstm_rmse > GARCH_RMSE and mlp_rmse > GARCH_RMSE
lstm_beats_mlp = lstm_rmse < mlp_rmse
beats_persistence_lstm = lstm_rmse < pers_rmse_check
beats_persistence_mlp = mlp_rmse < pers_rmse_check

# One denominator for the LSTM-vs-MLP gap, reused wherever it is quoted.
lstm_mlp_rel = abs(lstm_rmse - mlp_rmse) / max(lstm_rmse, mlp_rmse) * 100

best_nn_label = (f'LSTM ({LSTM_UNITS} units)' if lstm_beats_mlp
                 else f'MLP ({MLP_HIDDEN} hidden)')

print(f'LSTM vs GARCH: {lstm_garch_rel:.2e} relative '
      f'({"tie" if lstm_garch_tie else "material"})')
print(f'MLP vs GARCH:  {mlp_garch_rel:.2e} relative '
      f'({"tie" if mlp_garch_tie else "material"})')

if lstm_garch_tie:
    lstm_verdict = (
        f"On this seed the LSTM and {GARCH_LABEL} are level: RMSE differs by "
        f"{lstm_garch_rel:.2e} in relative terms ({lstm_rmse:.6f} against "
        f"{GARCH_RMSE:.6f}), and on MAE the LSTM is "
        f"{'lower' if lstm_mae < GARCH_MAE else 'higher'} "
        f"({lstm_mae:.6f} against {GARCH_MAE:.6f})."
    )
    cost_note = (
        f" Parity is not an improvement, and it is bought at a price: "
        f"{GARCH_LABEL} reaches the same accuracy with {garch_n_params} "
        f"parameters and no training loop, against {lstm_params:,} parameters "
        f"and {wf_total_lstm:.0f} seconds across {n_blocks} refits."
    )
elif lstm_rmse > GARCH_RMSE:
    lstm_verdict = (
        f"On this seed {GARCH_LABEL} beat the LSTM by "
        f"{lstm_garch_rel * 100:.1f}% on RMSE "
        f"({GARCH_RMSE:.6f} against {lstm_rmse:.6f})."
    )
    cost_note = (
        f" The margin comes with a cost asymmetry: GARCH used "
        f"{garch_n_params} parameters and no training loop, the LSTM "
        f"{lstm_params:,} parameters and {wf_total_lstm:.0f} seconds."
    )
else:
    lstm_verdict = (
        f"On this seed the LSTM beat {GARCH_LABEL} by "
        f"{lstm_garch_rel * 100:.1f}% on RMSE "
        f"({lstm_rmse:.6f} against {GARCH_RMSE:.6f})."
    )
    cost_note = (
        f" Whether that margin justifies {lstm_params:,} parameters and "
        f"{wf_total_lstm:.0f} seconds of training is a deployment question, "
        f"and section 10 shows the margin is not stable across seeds."
    )

if mlp_garch_tie:
    mlp_verdict = f"The MLP is level with {GARCH_LABEL}."
elif mlp_rmse > GARCH_RMSE:
    mlp_verdict = (
        f"The MLP fell short by {mlp_garch_rel * 100:.1f}%"
        + (f", and did not beat persistence either ({mlp_rmse:.6f} against "
           f"{pers_rmse_check:.6f})." if not beats_persistence_mlp else ".")
    )
else:
    mlp_verdict = f"The MLP beat {GARCH_LABEL} by {mlp_garch_rel * 100:.1f}%."

display(Markdown(f"""
### Interpretation

{lstm_verdict} {mlp_verdict}{cost_note}

The LSTM {'outperformed' if lstm_beats_mlp else 'underperformed'} the MLP by
{lstm_mlp_rel:.1f}% on RMSE. Whether that gap is distinguishable from noise is
tested below rather than assumed here.

Two caveats govern everything in this section. It is a single
{TEST_SIZE}-day test window, so the ranking could differ under another market
regime. And it is a single seed: section 10 measured a
{lstm_spread_pct:.1f}% spread in LSTM RMSE across {len(SEED_LIST)} otherwise
identical runs, so the figures above should be read against that spread rather
than at face value.
"""))
```

    LSTM vs GARCH: 2.88e-02 relative (material)
    MLP vs GARCH:  4.49e-01 relative (material)
    



### Interpretation

On this seed GJR-GARCH(1,1,1) — skewed Student's t beat the LSTM by 2.9% on RMSE (0.005236 against 0.005387). The MLP fell short by 44.9%, and did not beat persistence either (0.007588 against 0.007562). The margin comes with a cost asymmetry: GARCH used 6 parameters and no training loop, the LSTM 5,537 parameters and 684 seconds.

The LSTM outperformed the MLP by
29.0% on RMSE. Whether that gap is distinguishable from noise is
tested below rather than assumed here.

Two caveats govern everything in this section. It is a single
252-day test window, so the ranking could differ under another market
regime. And it is a single seed: section 10 measured a
1.6% spread in LSTM RMSE across 3 otherwise
identical runs, so the figures above should be read against that spread rather
than at face value.



### Bootstrap confidence intervals

A single RMSE figure gives no sense of sampling uncertainty. The intervals
below use a moving block bootstrap rather than an independent resample.
Forecast errors in volatility models are autocorrelated, the same property
that motivates the Newey-West correction in the Diebold-Mariano test, and
resampling individual days independently would break that dependence and
understate the interval width. Blocks of 21 days preserve the local error
structure.

This measures a different uncertainty from section 10. The bootstrap asks how
much the RMSE of *this* forecast series would move under a different draw of
test days. The seed spread asks how much the forecast series itself moves when
the model is refitted. Both are real, and they compound.


```python
def block_bootstrap_rmse_ci(errors, n_boot=N_BOOT, block=BOOT_BLOCK,
                            alpha=0.05, seed=SEED):
    """Moving block bootstrap confidence interval for RMSE."""
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors)
    n = len(errors)
    n_blk = int(np.ceil(n / block))
    max_start = n - block + 1

    boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, n_blk)
        sample = np.concatenate([errors[s : s + block] for s in starts])[:n]
        boot[b] = np.sqrt(np.mean(sample ** 2))

    return tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


lstm_ci = block_bootstrap_rmse_ci(lstm_errors)
mlp_ci = block_bootstrap_rmse_ci(mlp_errors)
pers_ci = block_bootstrap_rmse_ci(pers_errors)

ci_overlap_lstm_mlp = not (lstm_ci[1] < mlp_ci[0] or mlp_ci[1] < lstm_ci[0])
garch_in_lstm_ci = lstm_ci[0] <= GARCH_RMSE <= lstm_ci[1]

display(Markdown(f"""
| Model | RMSE | 95% CI (block bootstrap) | Width |
|---|---|---|---|
| Persistence | {pers_rmse_check:.6f} | [{pers_ci[0]:.6f}, {pers_ci[1]:.6f}] | {pers_ci[1] - pers_ci[0]:.6f} |
| {GARCH_LABEL} | {GARCH_RMSE:.6f} | (not computed — see NB05) | — |
| MLP ({MLP_HIDDEN} hidden) | {mlp_rmse:.6f} | [{mlp_ci[0]:.6f}, {mlp_ci[1]:.6f}] | {mlp_ci[1] - mlp_ci[0]:.6f} |
| LSTM ({LSTM_UNITS} units) | {lstm_rmse:.6f} | [{lstm_ci[0]:.6f}, {lstm_ci[1]:.6f}] | {lstm_ci[1] - lstm_ci[0]:.6f} |

Based on {N_BOOT:,} bootstrap replications with {BOOT_BLOCK}-day blocks, on the
primary seed ({SEED}).

The LSTM and MLP intervals
{'overlap, so their difference is not distinguishable from sampling noise on this window.' if ci_overlap_lstm_mlp else 'do not overlap, so the difference between them is unlikely to be sampling noise alone.'}
The GARCH RMSE
{'falls inside the LSTM interval, so the two are statistically indistinguishable on this evidence.' if garch_in_lstm_ci else 'falls outside the LSTM interval, which supports treating the difference as real rather than incidental.'}

Computing an equivalent interval for GARCH requires its walk-forward error
series from Notebook 05.
"""))
```



| Model | RMSE | 95% CI (block bootstrap) | Width |
|---|---|---|---|
| Persistence | 0.007562 | [0.006327, 0.008864] | 0.002536 |
| GJR-GARCH(1,1,1) — skewed Student's t | 0.005236 | (not computed — see NB05) | — |
| MLP (64 hidden) | 0.007588 | [0.006613, 0.008673] | 0.002060 |
| LSTM (32 units) | 0.005387 | [0.004577, 0.006206] | 0.001629 |

Based on 2,000 bootstrap replications with 21-day blocks, on the
primary seed (42).

The LSTM and MLP intervals
do not overlap, so the difference between them is unlikely to be sampling noise alone.
The GARCH RMSE
falls inside the LSTM interval, so the two are statistically indistinguishable on this evidence.

Computing an equivalent interval for GARCH requires its walk-forward error
series from Notebook 05.



### Diebold-Mariano tests


```python
def diebold_mariano_hac(e1, e2, h=1):
    """Diebold-Mariano test with Newey-West HAC variance estimator.

    H0: E[L(e1) - L(e2)] = 0 where L is squared error.
    A positive statistic means the second model is more accurate.
    """
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    d_bar = d.mean()
    max_lag = max(h - 1, 1)

    gamma_0 = np.sum((d - d_bar) ** 2) / T
    gamma_sum = 0
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        gamma_sum += 2 * weight * gamma_k

    hac_var = (gamma_0 + gamma_sum) / T
    if hac_var <= 0:
        hac_var = np.var(d, ddof=1) / T

    dm_stat = d_bar / np.sqrt(hac_var)
    return dm_stat, 2 * stats.norm.sf(abs(dm_stat))


dm_lp, p_lp = diebold_mariano_hac(pers_errors, lstm_errors)
dm_mp, p_mp = diebold_mariano_hac(pers_errors, mlp_errors)
dm_lm, p_lm = diebold_mariano_hac(mlp_errors, lstm_errors)

display(Markdown(f"""
| Comparison | DM statistic | p-value | Significant at 5%? |
|---|---|---|---|
| LSTM vs Persistence | {dm_lp:.3f} | {p_lp:.4f} | {'Yes' if p_lp < 0.05 else 'No'} |
| MLP vs Persistence | {dm_mp:.3f} | {p_mp:.4f} | {'Yes' if p_mp < 0.05 else 'No'} |
| LSTM vs MLP | {dm_lm:.3f} | {p_lm:.4f} | {'Yes' if p_lm < 0.05 else 'No'} |

The variance estimator uses a Newey-West (Bartlett kernel) correction for
autocorrelation in the loss differential. A direct neural-network-versus-GARCH
test requires the GARCH forecast series from Notebook 05, which is not
currently exported.
"""))
```



| Comparison | DM statistic | p-value | Significant at 5%? |
|---|---|---|---|
| LSTM vs Persistence | 5.736 | 0.0000 | Yes |
| MLP vs Persistence | -0.080 | 0.9361 | No |
| LSTM vs MLP | 8.339 | 0.0000 | Yes |

The variance estimator uses a Newey-West (Bartlett kernel) correction for
autocorrelation in the loss differential. A direct neural-network-versus-GARCH
test requires the GARCH forecast series from Notebook 05, which is not
currently exported.



## 12. Computational cost


```python
X_test_full = X_seq[test_start : test_start + TEST_SIZE]
X_test_full_s = last_scaler.transform(
    X_test_full.reshape(-1, n_features)
).reshape(X_test_full.shape)

_ = last_lstm.predict(X_test_full_s[:1], verbose=0)
_ = last_mlp.predict(X_test_full_s[:1], verbose=0)

t0 = time.perf_counter()
_ = last_lstm.predict(X_test_full_s, verbose=0)
lstm_inference_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
_ = last_mlp.predict(X_test_full_s, verbose=0)
mlp_inference_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
_ = y_seq[test_start - 1 : test_start + TEST_SIZE - 1].copy()
pers_inference_ms = (time.perf_counter() - t0) * 1000

display(Markdown(f"""
### Training cost

| Model | Per seed | All {len(SEED_LIST)} seeds | Avg epochs |
|---|---|---|---|
| Persistence | 0 s | 0 s | — |
| {GARCH_LABEL} | (see NB05) | (deterministic, one fit suffices) | — |
| MLP ({MLP_HIDDEN} hidden) | {wf_total_mlp:.0f} s | {sum(runs[s]['time_mlp'] for s in SEED_LIST):.0f} s | {np.mean(epochs_used_mlp):.0f} |
| LSTM ({LSTM_UNITS} units) | {wf_total_lstm:.0f} s | {sum(runs[s]['time_lstm'] for s in SEED_LIST):.0f} s | {np.mean(epochs_used_lstm):.0f} |

Total wall clock for the walk-forward across all seeds: {wf_total:.0f} s.

### Inference speed ({TEST_SIZE} observations)

| Model | Time |
|---|---|
| Persistence | {pers_inference_ms:.2f} ms |
| {GARCH_LABEL} | (see NB05) |
| MLP ({MLP_HIDDEN} hidden) | {mlp_inference_ms:.1f} ms |
| LSTM ({LSTM_UNITS} units) | {lstm_inference_ms:.1f} ms |

The multi-seed column is the honest training cost. A model that must be run
several times before its output can be trusted costs several runs, and GARCH
needs one fit because repeating it changes nothing.
"""))
```



### Training cost

| Model | Per seed | All 3 seeds | Avg epochs |
|---|---|---|---|
| Persistence | 0 s | 0 s | — |
| GJR-GARCH(1,1,1) — skewed Student's t | (see NB05) | (deterministic, one fit suffices) | — |
| MLP (64 hidden) | 237 s | 686 s | 37 |
| LSTM (32 units) | 684 s | 1871 s | 42 |

Total wall clock for the walk-forward across all seeds: 2575 s.

### Inference speed (252 observations)

| Model | Time |
|---|---|
| Persistence | 0.09 ms |
| GJR-GARCH(1,1,1) — skewed Student's t | (see NB05) |
| MLP (64 hidden) | 93.2 ms |
| LSTM (32 units) | 156.8 ms |

The multi-seed column is the honest training cost. A model that must be run
several times before its output can be trusted costs several runs, and GARCH
needs one fit because repeating it changes nothing.



## 13. Diagnostics

All diagnostics below use the primary seed. Section 10 established that another
seed would move these figures, in some cases substantially.

### Training curves


```python
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('LSTM (first and last refit)',
                                    'MLP (first and last refit)'))

for col, (histories, label) in enumerate([
    (training_histories_lstm, 'LSTM'),
    (training_histories_mlp, 'MLP'),
], start=1):
    for idx, block_label in [(0, 'Block 1'), (-1, f'Block {n_blocks}')]:
        h = histories[idx]
        fig.add_trace(go.Scatter(
            y=h['loss'], name=f'{block_label} — train', mode='lines',
            line=dict(dash='solid'), legendgroup=label,
            showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            y=h['val_loss'], name=f'{block_label} — val', mode='lines',
            line=dict(dash='dash'), legendgroup=label,
            showlegend=(col == 1),
        ), row=1, col=col)

fig.update_layout(template='plotly_dark',
                  title='Training curves (first and last refit)', height=400)
fig.update_xaxes(title_text='Epoch')
fig.update_yaxes(title_text='MSE loss')
fig.show()

first_stopped = epochs_used_lstm[0]
last_stopped = epochs_used_lstm[-1]
last_train_size = test_start + (n_blocks - 1) * REFIT_EVERY
samples_per_param = last_train_size / lstm_params

display(Markdown(f"""
LSTM early stopping triggered at epoch {first_stopped} (first block) and
epoch {last_stopped} (last block).
{'Validation loss stabilised before the epoch ceiling, suggesting the network reached its capacity limit rather than exhausting the training budget.' if max(first_stopped, last_stopped) < EPOCHS - PATIENCE else 'The network used most of its epoch budget, suggesting it was still learning when stopped.'}

The last block trained on {last_train_size:,} sequences against
{lstm_params:,} parameters, a ratio of {samples_per_param:.1f}:1.
{'That is comfortable.' if samples_per_param >= 20 else 'Deep learning generally wants 100:1 or more, so this is the data-limited regime, and it is the most likely source of the seed instability measured in section 10.'}
"""))
```





LSTM early stopping triggered at epoch 16 (first block) and
epoch 16 (last block).
Validation loss stabilised before the epoch ceiling, suggesting the network reached its capacity limit rather than exhausting the training budget.

The last block trained on 6,399 sequences against
5,537 parameters, a ratio of 1.2:1.
Deep learning generally wants 100:1 or more, so this is the data-limited regime, and it is the most likely source of the seed instability measured in section 10.



### Forecast comparison


```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_dates, y=actual, name='Actual |return|',
                         mode='lines', line=dict(color='white', width=1),
                         opacity=0.5))
fig.add_trace(go.Scatter(x=pred_dates, y=lstm_forecasts,
                         name=f'LSTM ({LSTM_UNITS} units)', mode='lines',
                         line=dict(color='#00d4aa', width=1.5)))
fig.add_trace(go.Scatter(x=pred_dates, y=mlp_forecasts,
                         name=f'MLP ({MLP_HIDDEN} hidden)', mode='lines',
                         line=dict(color='#ffd700', width=1.5)))
fig.add_trace(go.Scatter(x=pred_dates, y=persistence, name='Persistence',
                         mode='lines',
                         line=dict(color='#ff6b6b', width=1, dash='dot'),
                         opacity=0.6))
fig.update_layout(
    template='plotly_dark',
    title=f'Walk-forward forecasts (seed {SEED})',
    xaxis_title='Date', yaxis_title='|Daily log return|', height=450,
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
)
fig.show()
```




```python
# Every seed's LSTM forecast on one axis: the spread made visible.
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_dates, y=actual, name='Actual |return|',
                         mode='lines', line=dict(color='white', width=1),
                         opacity=0.4))
for seed in SEED_LIST:
    fig.add_trace(go.Scatter(
        x=pred_dates, y=runs[seed]['lstm_forecasts'],
        name=f'LSTM seed {seed}', mode='lines', line=dict(width=1.2),
    ))
fig.update_layout(
    template='plotly_dark',
    title='LSTM forecasts across seeds (identical data, code and hardware)',
    xaxis_title='Date', yaxis_title='|Daily log return|', height=450,
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
)
fig.show()

display(Markdown("""
The forecast paths, not just their error totals, differ across seeds. Where
they diverge most is where the risk report would have said different things
on the same day.
"""))
```





The forecast paths, not just their error totals, differ across seeds. Where
they diverge most is where the risk report would have said different things
on the same day.



### Residual autocorrelation

Significant autocorrelation means the network left predictable structure in
its errors: a better model could have used yesterday's error to improve
today's forecast. No significant autocorrelation means the remaining error is
closer to white noise, and further gains would require new information rather
than better use of the existing sequence.


```python
lb_rows = []
for label, errs in [
    (f'LSTM ({LSTM_UNITS} units)', lstm_errors),
    (f'MLP ({MLP_HIDDEN} hidden)', mlp_errors),
    ('Persistence', pers_errors),
]:
    lb = acorr_ljungbox(errs, lags=LB_LAGS, return_df=True)
    for lag in LB_LAGS:
        lb_rows.append({
            'Model': label, 'Lags': lag,
            'LB statistic': lb.loc[lag, 'lb_stat'],
            'p-value': lb.loc[lag, 'lb_pvalue'],
            'Autocorrelated at 5%?':
                'Yes' if lb.loc[lag, 'lb_pvalue'] < 0.05 else 'No',
        })

lb_df = pd.DataFrame(lb_rows)
lb_fmt = lb_df.copy()
lb_fmt['LB statistic'] = lb_fmt['LB statistic'].map('{:.2f}'.format)
lb_fmt['p-value'] = lb_fmt['p-value'].map('{:.4g}'.format)
display(Markdown(lb_fmt.to_markdown(index=False)))

lstm_lb_p = lb_df[(lb_df['Model'].str.startswith('LSTM'))
                  & (lb_df['Lags'] == LB_LAGS[-1])]['p-value'].iloc[0]
mlp_lb_p = lb_df[(lb_df['Model'].str.startswith('MLP'))
                 & (lb_df['Lags'] == LB_LAGS[-1])]['p-value'].iloc[0]

if lstm_lb_p < 0.05:
    lb_interp = (
        f'LSTM residuals show significant autocorrelation at {LB_LAGS[-1]} '
        f'lags (p = {lstm_lb_p:.4g}). Predictable temporal structure remains '
        f'in the errors, so this fit did not extract what was available from '
        f'the sequence it was given.'
    )
else:
    lb_interp = (
        f'LSTM residuals show no significant autocorrelation at '
        f'{LB_LAGS[-1]} lags (p = {lstm_lb_p:.4g}). The remaining error '
        f'behaves like white noise on this seed.'
    )

lb_interp += (
    f' MLP residuals {"are" if mlp_lb_p < 0.05 else "are not"} significantly '
    f'autocorrelated (p = {mlp_lb_p:.4g}).'
)

display(Markdown(lb_interp))
```


| Model           |   Lags |   LB statistic |   p-value | Autocorrelated at 5%?   |
|:----------------|-------:|---------------:|----------:|:------------------------|
| LSTM (32 units) |     10 |           5.44 | 0.86      | No                      |
| LSTM (32 units) |     21 |          10.07 | 0.978     | No                      |
| MLP (64 hidden) |     10 |         101.79 | 2.388e-17 | Yes                     |
| MLP (64 hidden) |     21 |         107.61 | 1.272e-13 | Yes                     |
| Persistence     |     10 |          73.58 | 8.974e-12 | Yes                     |
| Persistence     |     21 |          80.67 | 6.227e-09 | Yes                     |



LSTM residuals show no significant autocorrelation at 21 lags (p = 0.978). The remaining error behaves like white noise on this seed. MLP residuals are significantly autocorrelated (p = 1.272e-13).


### Forecast calibration

A calibration plot asks a question RMSE cannot: across the range of
predictions, are they right on average? Perfect calibration puts every point
on the 45 degree line. The fitted slope summarises the departure: below 1 means
forecasts are too spread out relative to reality, above 1 means too compressed.


```python
def calibration_data(forecast, actual_vals, n_bins=10):
    bins = pd.qcut(forecast, n_bins, labels=False, duplicates='drop')
    binned = pd.DataFrame({'pred': forecast, 'act': actual_vals, 'bin': bins})
    grouped = binned.groupby('bin', observed=True).mean()
    slope, intercept, r_val, _, _ = stats.linregress(forecast, actual_vals)
    return grouped['pred'].values, grouped['act'].values, slope, intercept, r_val


fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'LSTM ({LSTM_UNITS} units)',
                                    f'MLP ({MLP_HIDDEN} hidden)'))

cal_summary = {}
for col, (fc, label, colour) in enumerate([
    (lstm_forecasts, 'LSTM', '#00d4aa'),
    (mlp_forecasts, 'MLP', '#ffd700'),
], start=1):
    bp, ba, slope, intercept, r_val = calibration_data(fc, actual)
    cal_summary[label] = {'slope': slope, 'intercept': intercept, 'r': r_val}

    fig.add_trace(go.Scatter(x=fc, y=actual, mode='markers',
                             marker=dict(color=colour, size=3, opacity=0.3),
                             showlegend=False), row=1, col=col)
    fig.add_trace(go.Scatter(x=bp, y=ba, mode='markers+lines',
                             marker=dict(color='white', size=9,
                                         symbol='diamond'),
                             line=dict(color='white', width=2),
                             name='Binned mean',
                             showlegend=(col == 1)), row=1, col=col)
    lim = max(fc.max(), actual.max())
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode='lines',
                             line=dict(color='#ff6b6b', dash='dash',
                                       width=1.5),
                             name='Perfect calibration',
                             showlegend=(col == 1)), row=1, col=col)

fig.update_layout(template='plotly_dark', height=420,
                  title_text='Calibration: predicted versus actual |return|',
                  legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
fig.update_xaxes(title_text='Predicted |return|')
fig.update_yaxes(title_text='Actual |return|')
fig.show()

cal_lines = []
for label, s in cal_summary.items():
    direction = ('too dispersed relative to realised values'
                 if s['slope'] < 0.9 else
                 'too compressed relative to realised values'
                 if s['slope'] > 1.1 else 'close to proportional')
    cal_lines.append(
        f"- **{label}**: slope {s['slope']:.3f}, "
        f"intercept {s['intercept']:.6f}, r = {s['r']:.3f}. "
        f"Forecasts are {direction}."
    )

display(Markdown('Fitted calibration lines (seed ' + str(SEED) + '):\n\n'
                 + '\n'.join(cal_lines)))
```




Fitted calibration lines (seed 42):

- **LSTM**: slope 0.724, intercept 0.002624, r = 0.164. Forecasts are too dispersed relative to realised values.
- **MLP**: slope 0.043, intercept 0.006145, r = 0.028. Forecasts are too dispersed relative to realised values.


### Regime-conditional performance


```python
vol_annual = df['vol_roll_21'] * np.sqrt(252)
p25, p75, p95 = vol_annual.quantile([0.25, 0.75, 0.95])

regime_labels = pd.cut(vol_annual, bins=[-np.inf, p25, p75, p95, np.inf],
                       labels=['Calm', 'Normal', 'Stress', 'Crisis'])
test_regimes = regime_labels.loc[pred_dates].values

regime_results = []
for regime in ['Calm', 'Normal', 'Stress', 'Crisis']:
    mask = test_regimes == regime
    n = mask.sum()
    if n < 5:
        continue
    a = actual[mask]
    regime_results.append({
        'Regime': regime, 'Days': int(n),
        'Persistence RMSE': root_mean_squared_error(a, persistence[mask]),
        'MLP RMSE': root_mean_squared_error(a, mlp_forecasts[mask]),
        'LSTM RMSE': root_mean_squared_error(a, lstm_forecasts[mask]),
    })

regime_df = pd.DataFrame(regime_results)
regime_df['LSTM vs Pers'] = (
    (regime_df['Persistence RMSE'] - regime_df['LSTM RMSE'])
    / regime_df['Persistence RMSE'] * 100).round(1).astype(str) + '%'
regime_df['MLP vs Pers'] = (
    (regime_df['Persistence RMSE'] - regime_df['MLP RMSE'])
    / regime_df['Persistence RMSE'] * 100).round(1).astype(str) + '%'

display(regime_df)

small_regimes = regime_df[regime_df['Days'] < 30]['Regime'].tolist()
display(Markdown(f"""
Regime labels here come from 21-day realised volatility percentiles rather
than GJR-GARCH conditional volatility, so they will not match NB05's labels
exactly.
{'Note the sample sizes: ' + ', '.join(f"{r} has fewer than 30 days" for r in small_regimes) + '. Percentage comparisons on that few observations carry very wide error bars and should not be read as regime-specific findings.' if small_regimes else ''}
"""))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regime</th>
      <th>Days</th>
      <th>Persistence RMSE</th>
      <th>MLP RMSE</th>
      <th>LSTM RMSE</th>
      <th>LSTM vs Pers</th>
      <th>MLP vs Pers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Calm</td>
      <td>40</td>
      <td>0.003672</td>
      <td>0.004596</td>
      <td>0.003411</td>
      <td>7.1%</td>
      <td>-25.2%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Normal</td>
      <td>205</td>
      <td>0.007900</td>
      <td>0.007882</td>
      <td>0.005590</td>
      <td>29.2%</td>
      <td>0.2%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stress</td>
      <td>7</td>
      <td>0.012389</td>
      <td>0.011533</td>
      <td>0.007950</td>
      <td>35.8%</td>
      <td>6.9%</td>
    </tr>
  </tbody>
</table>
</div>




Regime labels here come from 21-day realised volatility percentiles rather
than GJR-GARCH conditional volatility, so they will not match NB05's labels
exactly.
Note the sample sizes: Stress has fewer than 30 days. Percentage comparisons on that few observations carry very wide error bars and should not be read as regime-specific findings.



### The largest forecasting miss


```python
worst_idx = int(np.argmax(np.abs(lstm_errors)))
worst_date = pred_dates[worst_idx]
worst_actual = actual[worst_idx]
worst_lstm = lstm_forecasts[worst_idx]
worst_mlp = mlp_forecasts[worst_idx]
worst_pers = persistence[worst_idx]
worst_err = lstm_errors[worst_idx]
worst_regime = test_regimes[worst_idx]

lb_window = 5
trailing = actual[max(worst_idx - lb_window, 0):worst_idx]
trailing_mean = trailing.mean() if len(trailing) else np.nan
spike_ratio = worst_actual / trailing_mean if trailing_mean else np.nan

display(Markdown(f"""
| | Value |
|---|---|
| Date | {worst_date.date()} |
| Regime | {worst_regime} |
| Actual \\|return\\| | {worst_actual:.6f} |
| LSTM forecast | {worst_lstm:.6f} |
| MLP forecast | {worst_mlp:.6f} |
| Persistence forecast | {worst_pers:.6f} |
| LSTM error | {worst_err:+.6f} |
| Trailing {lb_window}-day mean | {trailing_mean:.6f} |
| Ratio to trailing mean | {spike_ratio:.2f}x |
"""))

if worst_err > 0:
    miss_type = 'underforecast'
    miss_expl = (
        f'The realised move was {spike_ratio:.1f} times the trailing '
        f'{lb_window}-day average. Every input the model had described the '
        f'market as it was before the shock, not as it became. A shock '
        f'arriving from outside the return series is not forecastable from '
        f'the return series.'
    )
else:
    miss_type = 'overforecast'
    miss_expl = (
        f'The model expected elevated volatility and the market delivered a '
        f'quiet day, realised {worst_actual:.6f} against a forecast of '
        f'{worst_lstm:.6f}. Overforecasting on quiet days is the signature of '
        f'insufficient mean reversion: GARCH encodes the decay rate '
        f'explicitly in its beta parameter, while the network has to learn it '
        f'from examples, and section 10 shows how much that learned decay '
        f'varies between runs.'
    )

display(Markdown(f"""
The worst miss was an **{miss_type}** of {abs(worst_err):.6f} on
{worst_date.date()}. {miss_expl}

Persistence forecast {worst_pers:.6f} against a realised {worst_actual:.6f}, so
the benchmark
{'also missed badly' if abs(worst_actual - worst_pers) > abs(worst_err) * 0.8 else 'was closer on this particular day'}.
Single-day failures are expected; the diagnostic question is whether the
failure mode is systematic, which the calibration plot and regime table
address.
"""))
```



| | Value |
|---|---|
| Date | 2026-06-05 |
| Regime | Normal |
| Actual \|return\| | 0.026801 |
| LSTM forecast | 0.003985 |
| MLP forecast | 0.000812 |
| Persistence forecast | 0.004047 |
| LSTM error | +0.022817 |
| Trailing 5-day mean | 0.003506 |
| Ratio to trailing mean | 7.64x |





The worst miss was an **underforecast** of 0.022817 on
2026-06-05. The realised move was 7.6 times the trailing 5-day average. Every input the model had described the market as it was before the shock, not as it became. A shock arriving from outside the return series is not forecastable from the return series.

Persistence forecast 0.004047 against a realised 0.026801, so
the benchmark
also missed badly.
Single-day failures are expected; the diagnostic question is whether the
failure mode is systematic, which the calibration plot and regime table
address.



### Feature importance (permutation)


```python
base_preds = last_lstm.predict(X_test_full_s, verbose=0).flatten()
base_mse = np.mean((actual - base_preds) ** 2)

N_REPEATS = 5
rng_perm = np.random.default_rng(SEED)
importance_scores, importance_std = {}, {}

for feat_idx, feat_name in enumerate(FEATURE_COLS):
    deltas = []
    for _ in range(N_REPEATS):
        X_perm = X_test_full_s.copy()
        flat = X_perm[:, :, feat_idx].flatten()
        rng_perm.shuffle(flat)
        X_perm[:, :, feat_idx] = flat.reshape(X_perm.shape[0], X_perm.shape[1])
        perm_preds = last_lstm.predict(X_perm, verbose=0).flatten()
        deltas.append(np.mean((actual - perm_preds) ** 2) - base_mse)
    importance_scores[feat_name] = np.mean(deltas)
    importance_std[feat_name] = np.std(deltas, ddof=1)

imp_df = pd.DataFrame({
    'MSE increase': pd.Series(importance_scores),
    'Std across repeats': pd.Series(importance_std),
}).sort_values('MSE increase', ascending=False)
imp_df['Signal / noise'] = (
    imp_df['MSE increase'] / imp_df['Std across repeats'].replace(0, np.nan)
).round(2)

display(imp_df)

top_feature = imp_df.index[0]
print(f'\nMost important feature (seed {SEED}): {top_feature}')

# What each input adds relative to GARCH, whose only input is the daily return
# series. Log returns, rolling volatility with its rank and ratio, RSI and
# Bollinger width are all built from closing prices, so they re-express
# information GARCH already models. ATR, which uses the intraday high and low,
# and volume are the only inputs carrying information GARCH never sees.
from_closes = {'log_returns', 'vol_roll_10', 'vol_roll_21', 'vol_roll_60',
               'vol_rank_30', 'vol_ratio_10_60', 'rsi_14', 'bb_width'}
outside_garch = {'atr_pct_14', 'volume_rel_21'}
assert from_closes | outside_garch == set(FEATURE_COLS), (
    'classify every feature as built from closing prices or outside GARCH')

top_3 = list(imp_df.index[:3])
top_outside = [f for f in top_3 if f in outside_garch]
top_closes = [f for f in top_3 if f in from_closes]
n_negative = int((imp_df['MSE increase'] < 0).sum())

def join_names(names):
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]

if top_outside:
    feat_interp = (
        f"The top three features are {join_names(top_3)}. Only "
        f"{join_names(top_outside)} {'carries' if len(top_outside) == 1 else 'carry'} "
        f"information GARCH never sees; {join_names(top_closes)} "
        f"{'is' if len(top_closes) == 1 else 'are'} built from the same closing "
        f"prices GARCH models."
    )
    # Quoted again in section 15.
    importance_summary = (
        f"puts {join_names(top_outside)}, which GARCH never sees, among the "
        f"three inputs the LSTM relies on most, so it is not only re-learning "
        f"the return series"
    )
else:
    feat_interp = (
        f"The top three features are {join_names(top_3)}, all built from the "
        f"closing prices GARCH models, so on this seed the network re-expresses "
        f"information GARCH already uses, at far higher computational cost."
    )
    importance_summary = (
        "ranks only inputs built from closing prices among the three the LSTM "
        "relies on most, so on this seed it appears to re-learn information "
        "GARCH already uses"
    )

display(Markdown(
    feat_interp +
    f" {n_negative} of {len(FEATURE_COLS)} features show negative importance, "
    "meaning shuffling them improved the forecast. On a well-fitted model that "
    "indicates noise features; on this one it is also consistent with the fit "
    "instability measured in section 10, and the ranking should be treated as "
    "specific to this seed rather than as a property of the architecture."
))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSE increase</th>
      <th>Std across repeats</th>
      <th>Signal / noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>log_returns</th>
      <td>7.399577e-07</td>
      <td>2.541274e-07</td>
      <td>2.91</td>
    </tr>
    <tr>
      <th>vol_roll_60</th>
      <td>5.218517e-07</td>
      <td>5.504214e-08</td>
      <td>9.48</td>
    </tr>
    <tr>
      <th>volume_rel_21</th>
      <td>4.763732e-07</td>
      <td>2.417227e-07</td>
      <td>1.97</td>
    </tr>
    <tr>
      <th>atr_pct_14</th>
      <td>1.250137e-07</td>
      <td>5.828038e-08</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>vol_roll_10</th>
      <td>9.269774e-08</td>
      <td>3.035954e-08</td>
      <td>3.05</td>
    </tr>
    <tr>
      <th>vol_rank_30</th>
      <td>7.202938e-08</td>
      <td>6.486483e-08</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>vol_roll_21</th>
      <td>-5.897114e-08</td>
      <td>2.781993e-08</td>
      <td>-2.12</td>
    </tr>
    <tr>
      <th>vol_ratio_10_60</th>
      <td>-1.271412e-07</td>
      <td>1.720177e-07</td>
      <td>-0.74</td>
    </tr>
    <tr>
      <th>bb_width</th>
      <td>-6.032977e-07</td>
      <td>9.092311e-08</td>
      <td>-6.64</td>
    </tr>
    <tr>
      <th>rsi_14</th>
      <td>-1.326490e-06</td>
      <td>1.166854e-07</td>
      <td>-11.37</td>
    </tr>
  </tbody>
</table>
</div>


    
    Most important feature (seed 42): log_returns
    


Non-volatility features (log_returns, volume_rel_21) rank in the top three, so the network is keying on information GARCH has no access to. That did not translate into a win here. 4 of 10 features show negative importance, meaning shuffling them improved the forecast. On a well-fitted model that indicates noise features; on this one it is also consistent with the fit instability measured in section 10, and the ranking should be treated as specific to this seed rather than as a property of the architecture.


### Complexity versus accuracy


```python
models = {
    'Persistence': (0, pers_rmse_check),
    GARCH_LABEL: (garch_n_params, GARCH_RMSE),
    f'MLP ({MLP_HIDDEN} hidden)': (mlp_params, mlp_rmse),
    f'LSTM ({LSTM_UNITS} units)': (lstm_params, lstm_rmse),
}

names = list(models.keys())
params_plot = [max(v[0], 1) for v in models.values()]
rmses = [v[1] for v in models.values()]

frontier = []
for i, (n_i, r_i) in enumerate(zip(params_plot, rmses)):
    dominated = any((n_j <= n_i and r_j < r_i)
                    for j, (n_j, r_j) in enumerate(zip(params_plot, rmses))
                    if j != i)
    if not dominated:
        frontier.append(i)

best_i = min(frontier, key=lambda i: rmses[i])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=params_plot, y=rmses, mode='markers+text', text=names,
    textposition='top center',
    marker=dict(size=14, color=['#ff6b6b', '#00d4aa', '#ffd700', '#00aaff']),
    textfont=dict(size=11),
))
# LSTM seed spread as a vertical bar: the point estimate is not the whole story.
fig.add_trace(go.Scatter(
    x=[lstm_params, lstm_params], y=[lstm_rmse_min, lstm_rmse_max],
    mode='lines', line=dict(color='#00aaff', width=3),
    name='LSTM seed range', showlegend=True,
))
fig.add_annotation(x=np.log10(params_plot[best_i]), y=rmses[best_i],
                   text='Best trade-off', showarrow=True, arrowhead=2,
                   arrowcolor='white', ax=45, ay=35,
                   font=dict(color='white', size=12),
                   bgcolor='rgba(0,0,0,0.55)', borderpad=4)
fig.update_layout(template='plotly_dark', title='Complexity vs accuracy',
                  xaxis_title='Trainable parameters (log scale)',
                  yaxis_title='RMSE', xaxis_type='log', height=450)
fig.show()

if lstm_garch_tie:
    complexity_note = (
        f'On this seed the LSTM matches GARCH at '
        f'{lstm_params / garch_n_params:,.0f} times the parameter count.'
    )
elif garch_wins:
    complexity_note = (
        'Neither network improved on GARCH on this seed, so the additional '
        'complexity bought nothing.'
    )
else:
    complexity_note = (
        'A network improved on GARCH on this seed, at substantial complexity '
        'cost.'
    )

display(Markdown(f"""
{GARCH_LABEL} achieves its accuracy with {garch_n_params} parameters. The MLP
uses {mlp_params:,} and the LSTM {lstm_params:,}. {complexity_note}

The vertical bar on the LSTM marker is its range across {len(SEED_LIST)} seeds.
GARCH has no such bar, because refitting it on the same data returns the same
answer.

The efficient frontier contains {len(frontier)} of the four models; the
remainder are dominated, meaning another model achieves lower RMSE with no
more parameters.
"""))
```





GJR-GARCH(1,1,1) — skewed Student's t achieves its accuracy with 6 parameters. The MLP
uses 13,569 and the LSTM 5,537. Neither network improved on GARCH on this seed, so the additional complexity bought nothing.

The vertical bar on the LSTM marker is its range across 3 seeds.
GARCH has no such bar, because refitting it on the same data returns the same
answer.

The efficient frontier contains 2 of the four models; the
remainder are dominated, meaning another model achieves lower RMSE with no
more parameters.



## 14. Deployment comparison


```python
garch_dominates = GARCH_RMSE <= min(lstm_rmse, mlp_rmse)
if garch_dominates:
    tradeoff_note = (
        "The MLP is the more reproducible of the two networks, which alone "
        "would count in its favour. It does not here: "
        f"{GARCH_LABEL} is both more accurate than either network and exactly "
        "reproducible, so it dominates on both axes and the trade-off between "
        "them never has to be priced."
    )
else:
    tradeoff_note = (
        "Accuracy and reproducibility point in different directions on this "
        "run, and which matters more is a deployment judgement rather than a "
        "statistical one."
    )

display(Markdown(f"""
| Criterion | {GARCH_LABEL} | MLP ({MLP_HIDDEN} hidden) | LSTM ({LSTM_UNITS} units) |
|---|---|---|---|
| RMSE (primary seed) | {GARCH_RMSE:.6f} | {mlp_rmse:.6f} | {lstm_rmse:.6f} |
| RMSE range across seeds | not applicable | {mlp_rmses.min():.6f} to {mlp_rmses.max():.6f} | {lstm_rmse_min:.6f} to {lstm_rmse_max:.6f} |
| Trainable parameters | {garch_n_params} | {mlp_params:,} | {lstm_params:,} |
| Training time | (see NB05) | {wf_total_mlp:.0f} s per seed | {wf_total_lstm:.0f} s per seed |
| Inference ({TEST_SIZE} obs) | (see NB05) | {mlp_inference_ms:.0f} ms | {lstm_inference_ms:.0f} ms |
| Interpretability | Each parameter maps to a financial mechanism | Black box | Black box |
| Dependencies | `arch` (built on NumPy and SciPy) | TensorFlow | TensorFlow |
| Reproducibility | Deterministic | {mlp_spread_pct:.0f}% RMSE spread across seeds | {lstm_spread_pct:.0f}% RMSE spread across seeds |

The reproducibility row is the one that decides Version 1. A daily risk report
has to be defensible when someone asks why today's number differs from
yesterday's, and a model whose output moves when nothing but the thread
schedule changed cannot answer that question. Accuracy alone would not settle
the choice between these models; auditability does.
"""))
```



| Criterion | GJR-GARCH(1,1,1) — skewed Student's t | MLP (64 hidden) | LSTM (32 units) |
|---|---|---|---|
| RMSE (primary seed) | 0.005236 | 0.007588 | 0.005387 |
| RMSE range across seeds | not applicable | 0.007547 to 0.008223 | 0.005387 to 0.005473 |
| Trainable parameters | 6 | 13,569 | 5,537 |
| Training time | (see NB05) | 237 s per seed | 684 s per seed |
| Inference (252 obs) | (see NB05) | 93 ms | 157 ms |
| Interpretability | Each parameter maps to a financial mechanism | Black box | Black box |
| Dependencies | `arch` (built on NumPy and SciPy) | TensorFlow | TensorFlow |
| Reproducibility | Deterministic | 9% RMSE spread across seeds | 2% RMSE spread across seeds |

The reproducibility row is the one that decides Version 1. A daily risk report
has to be defensible when someone asks why today's number differs from
yesterday's, and a model whose output moves when nothing but the thread
schedule changed cannot answer that question. Accuracy alone would not settle
the choice between these models; auditability does.



## 15. Why the structural model is retained


```python
if GARCH_PERSISTENCE is not None:
    half_life = np.log(0.5) / np.log(GARCH_PERSISTENCE)
    persistence_phrase = (
        f'GARCH persistence of {GARCH_PERSISTENCE:.4f} in Notebook 05, '
        f'implying a shock half-life of roughly {half_life:.0f} trading days'
    )
else:
    persistence_phrase = (
        'near-unit GARCH persistence in Notebook 05, implying shocks that '
        'decay over months rather than days'
    )

# The LSTM-vs-MLP claim is only as strong as the test behind it.
if p_lm < 0.05 and lstm_beats_mlp:
    seq_claim = (
        f'The LSTM beat the MLP by {lstm_mlp_rel:.1f}% and the difference is '
        f'significant (DM = {dm_lm:.3f}, p = {p_lm:.4f}). Since both see '
        f'identical inputs, the gap is attributable to the sequential '
        f'structure the MLP discards.'
    )
elif lstm_beats_mlp:
    seq_claim = (
        f'The LSTM scored {lstm_mlp_rel:.1f}% below the MLP on RMSE, but the '
        f'difference is not significant (DM = {dm_lm:.3f}, p = {p_lm:.4f}), '
        f'and the bootstrap intervals '
        f'{"overlap" if ci_overlap_lstm_mlp else "do not overlap"}. On this '
        f'evidence the second sub-hypothesis is unresolved: temporal '
        f'modelling cannot be shown to add value beyond the rolling features, '
        f'nor ruled out.'
    )
else:
    seq_claim = (
        f'The MLP matched or beat the LSTM (DM = {dm_lm:.3f}, '
        f'p = {p_lm:.4f}), suggesting the feature set carries whatever signal '
        f'exists and sequential modelling adds nothing.'
    )

display(Markdown(f"""
The result is not a failure of deep learning in general. It is a predictable
outcome of applying flexible models to a problem where the structure is
already known and the data is thin.

The comparison illustrates the bias-variance trade-off. GARCH restricts the
hypothesis space using financial assumptions: variance is persistent, shocks
decay geometrically, negative returns raise variance more than positive ones.
Those restrictions introduce bias if the assumptions are wrong, but they
collapse the estimation problem to {garch_n_params} parameters and keep
variance low. The networks have far greater representational capacity and
therefore lower approximation bias, but they need many more observations to
estimate their parameters reliably.

The variance term is not a theoretical concern here. It is measured. Section 10
found LSTM RMSE varying {lstm_spread_pct:.1f}% across {len(SEED_LIST)} runs
that differ only in seed. The networks trained on roughly {test_start:,}
sequences against {lstm_params:,} parameters (LSTM) and {mlp_params:,} (MLP),
ratios of ~{test_start / lstm_params:.1f}:1 and ~{test_start / mlp_params:.1f}:1.
Deep learning generally wants 100:1 or better. At these ratios the optimiser
has many near-equivalent solutions to choose between, and which one it lands on
depends on numerical accidents.

The target works against flexible models too. Absolute daily log returns are
dominated by idiosyncratic shocks, and the forecastable component, the
conditional variance, is a slow-moving signal embedded in fast noise. GARCH is
built to extract exactly that. A network has to learn to ignore the noise,
which takes more data than is available.

Volatility's high autocorrelation ({persistence_phrase}) makes the persistence
benchmark hard to beat and limits the room for any model to improve. GARCH's
edge comes from modelling mean reversion after shocks explicitly; the networks
must learn that decay from examples, and section 10 shows how unstably they
learn it.

{seq_claim}

Permutation importance suggests the LSTM keys on {top_feature} rather than on
the rolling volatility features, so it is not simply rediscovering GARCH.
Multi-asset inputs (cross-sectional volatility, sector correlations, VIX term
structure) could give deep learning a genuine information advantage in a future
version. On a single return series the structural model has fewer unknowns and
more constraints, and that is what decided this comparison.
"""))
```



The result is not a failure of deep learning in general. It is a predictable
outcome of applying flexible models to a problem where the structure is
already known and the data is thin.

The comparison illustrates the bias-variance trade-off. GARCH restricts the
hypothesis space using financial assumptions: variance is persistent, shocks
decay geometrically, negative returns raise variance more than positive ones.
Those restrictions introduce bias if the assumptions are wrong, but they
collapse the estimation problem to 6 parameters and keep
variance low. The networks have far greater representational capacity and
therefore lower approximation bias, but they need many more observations to
estimate their parameters reliably.

The variance term is not a theoretical concern here. It is measured. Section 10
found LSTM RMSE varying 1.6% across 3 runs
that differ only in seed. The networks trained on roughly 6,168
sequences against 5,537 parameters (LSTM) and 13,569 (MLP),
ratios of ~1.1:1 and ~0.5:1.
Deep learning generally wants 100:1 or better. At these ratios the optimiser
has many near-equivalent solutions to choose between, and which one it lands on
depends on numerical accidents.

The target works against flexible models too. Absolute daily log returns are
dominated by idiosyncratic shocks, and the forecastable component, the
conditional variance, is a slow-moving signal embedded in fast noise. GARCH is
built to extract exactly that. A network has to learn to ignore the noise,
which takes more data than is available.

Volatility's high autocorrelation (GARCH persistence of 0.9789 in Notebook 05, implying a shock half-life of roughly 33 trading days) makes the persistence
benchmark hard to beat and limits the room for any model to improve. GARCH's
edge comes from modelling mean reversion after shocks explicitly; the networks
must learn that decay from examples, and section 10 shows how unstably they
learn it.

The LSTM beat the MLP by 29.0% and the difference is significant (DM = 8.339, p = 0.0000). Since both see identical inputs, the gap is attributable to the sequential structure the MLP discards.

Permutation importance suggests the LSTM keys on log_returns rather than on
the rolling volatility features, so it is not simply rediscovering GARCH.
Multi-asset inputs (cross-sectional volatility, sector correlations, VIX term
structure) could give deep learning a genuine information advantage in a future
version. On a single return series the structural model has fewer unknowns and
more constraints, and that is what decided this comparison.



## 16. Conclusion


```python
if seeds_beating_garch == 0:
    headline = (
        f"Across {len(SEED_LIST)} seeds, no LSTM run improved on "
        f"{GARCH_LABEL}'s out-of-sample volatility forecasts, and the MLP "
        f"did not either."
    )
elif seeds_beating_garch == len(SEED_LIST):
    headline = (
        f"Across {len(SEED_LIST)} seeds, every LSTM run improved on "
        f"{GARCH_LABEL} on RMSE, though the margin varied by "
        f"{lstm_spread_pct:.1f}% between runs."
    )
else:
    headline = (
        f"Across {len(SEED_LIST)} seeds, {seeds_beating_garch} LSTM runs "
        f"improved on {GARCH_LABEL} and "
        f"{len(SEED_LIST) - seeds_beating_garch} did not. The comparison does "
        f"not have a stable answer at this sample size."
    )

p_lm_str = '< 0.001' if p_lm < 0.001 else f'= {p_lm:.4f}'
if not lstm_beats_mlp:
    mlp_control = f'matched or beat the LSTM (p {p_lm_str})'
elif p_lm < 0.05:
    mlp_control = (f'trailed the LSTM significantly '
                   f'(DM = {dm_lm:.3f}, p {p_lm_str})')
else:
    mlp_control = f'trailed the LSTM, though not significantly (p {p_lm_str})'

display(Markdown(f"""
{headline}

For this application, single-asset daily volatility on roughly {len(df):,}
observations, encoding financial structure proved more valuable than
increasing model capacity or enriching the feature set. Version 1 retains
{GARCH_LABEL} as the production forecasting model.

The strongest argument for that choice turned out not to be accuracy. It is
that GARCH returns the same coefficients every time it is fitted, while the
LSTM's walk-forward RMSE moved {lstm_spread_pct:.1f}% across runs differing
only in random seed. A daily risk report that changes when the pipeline is
re-run on unchanged data is not auditable, and auditability is a requirement
rather than a preference.

The MLP control answered its own question. On identical inputs it
{mlp_control}, so the feature set alone does not carry whatever signal exists.

Extensions for Version 2, out of current scope: multi-asset inputs;
Transformer or temporal convolutional architectures; GARCH-X with exogenous
features, to test whether GARCH also benefits from the Notebook 03 feature
set; and ensembling across seeds, which would trade compute for the stability
the single network lacks.

### Takeaway

Model complexity alone does not guarantee better forecasts. When the
underlying mechanism is well understood and the data is limited, explicitly
modelling that structure can match or beat far larger networks trained on
richer feature sets.

The less expected lesson is methodological. A single neural network run on
this data is not a measurement, it is a sample. Reporting one run as though it
were the model's performance would have produced a defensible-looking number
and a conclusion that flips depending on which execution gets written down.
The instability was only visible because the notebook was run twice, and it is
reported here because it is a finding rather than an inconvenience.
"""))
```



Across 3 seeds, no LSTM run improved on GJR-GARCH(1,1,1) — skewed Student's t's out-of-sample volatility forecasts, and the MLP did not either.

For this application, single-asset daily volatility on roughly 6,441
observations, encoding financial structure proved more valuable than
increasing model capacity or enriching the feature set. Version 1 retains
GJR-GARCH(1,1,1) — skewed Student's t as the production forecasting model.

The strongest argument for that choice turned out not to be accuracy. It is
that GARCH returns the same coefficients every time it is fitted, while the
LSTM's walk-forward RMSE moved 1.6% across runs differing
only in random seed. A daily risk report that changes when the pipeline is
re-run on unchanged data is not auditable, and auditability is a requirement
rather than a preference.

The MLP control answered its own question. On identical inputs it
trailed the LSTM significantly (DM = 8.339, p < 0.001), so the feature set alone does not carry whatever signal exists.

Extensions for Version 2, out of current scope: multi-asset inputs;
Transformer or temporal convolutional architectures; GARCH-X with exogenous
features, to test whether GARCH also benefits from the Notebook 03 feature
set; and ensembling across seeds, which would trade compute for the stability
the single network lacks.

### Takeaway

Model complexity alone does not guarantee better forecasts. When the
underlying mechanism is well understood and the data is limited, explicitly
modelling that structure can match or beat far larger networks trained on
richer feature sets.

The less expected lesson is methodological. A single neural network run on
this data is not a measurement, it is a sample. Reporting one run as though it
were the model's performance would have produced a defensible-looking number
and a conclusion that flips depending on which execution gets written down.
The instability was only visible because the notebook was run twice, and it is
reported here because it is a finding rather than an inconvenience.



## 17. Export


```python
preds_df = pd.DataFrame({
    'actual': actual,
    'persistence': persistence,
    'lstm': lstm_forecasts,
    'mlp': mlp_forecasts,
}, index=pred_dates)
for seed in SEED_LIST:
    preds_df[f'lstm_seed_{seed}'] = runs[seed]['lstm_forecasts']

preds_path = Path('../data/nb06_predictions.parquet')
preds_df.to_parquet(preds_path)
print(f'Predictions saved to {preds_path.resolve()}')

metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

metrics['notebook_06'] = {
    'primary_seed':           SEED,
    'seed_list':              list(SEED_LIST),
    'lstm_units':             LSTM_UNITS,
    'lstm_total_params':      int(lstm_params),
    'mlp_hidden':             MLP_HIDDEN,
    'mlp_total_params':       int(mlp_params),
    'garch_n_params':         int(garch_n_params),
    'n_features':             n_features,
    'lookback':               LOOKBACK,

    # Primary seed
    'wf_rmse_lstm':           float(lstm_rmse),
    'wf_mae_lstm':            float(lstm_mae),
    'wf_rmse_mlp':            float(mlp_rmse),
    'wf_mae_mlp':             float(mlp_mae),
    'wf_rmse_persistence':    float(pers_rmse_check),
    'wf_mae_persistence':     float(pers_mae_check),

    # Seed stability
    'lstm_rmse_by_seed':      {str(s): float(runs[s]['lstm_rmse'])
                               for s in SEED_LIST},
    'mlp_rmse_by_seed':       {str(s): float(runs[s]['mlp_rmse'])
                               for s in SEED_LIST},
    'lstm_rmse_mean':         float(lstm_rmse_mean),
    'lstm_rmse_sd':           float(lstm_rmse_sd),
    'lstm_rmse_min':          float(lstm_rmse_min),
    'lstm_rmse_max':          float(lstm_rmse_max),
    'lstm_spread_pct':        float(lstm_spread_pct),
    'mlp_spread_pct':         float(mlp_spread_pct),
    'seeds_beating_garch':    seeds_beating_garch,
    'seeds_beating_persistence': seeds_beating_pers,

    # Comparisons on the primary seed
    'lstm_vs_pers_rmse_pct':  float(lstm_vs_pers_rmse),
    'lstm_vs_garch_rmse_pct': float(lstm_vs_garch_rmse),
    'mlp_vs_pers_rmse_pct':   float(mlp_vs_pers_rmse),
    'mlp_vs_garch_rmse_pct':  float(mlp_vs_garch_rmse),
    'rmse_material_threshold': RMSE_MATERIAL,
    'lstm_garch_tie':         bool(lstm_garch_tie),
    'garch_wins':             bool(garch_wins),
    'lstm_beats_mlp':         bool(lstm_beats_mlp),

    # Tests
    'dm_stat_lstm_pers':      float(dm_lp),
    'dm_pval_lstm_pers':      float(p_lp),
    'dm_stat_mlp_pers':       float(dm_mp),
    'dm_pval_mlp_pers':       float(p_mp),
    'dm_stat_lstm_mlp':       float(dm_lm),
    'dm_pval_lstm_mlp':       float(p_lm),
    'rmse_ci_lstm':           [float(lstm_ci[0]), float(lstm_ci[1])],
    'rmse_ci_mlp':            [float(mlp_ci[0]), float(mlp_ci[1])],
    'rmse_ci_persistence':    [float(pers_ci[0]), float(pers_ci[1])],
    'n_bootstrap':            N_BOOT,
    'bootstrap_block':        BOOT_BLOCK,
    'lb_pval_lstm':           float(lstm_lb_p),
    'lb_pval_mlp':            float(mlp_lb_p),
    'calib_slope_lstm':       float(cal_summary['LSTM']['slope']),
    'calib_slope_mlp':        float(cal_summary['MLP']['slope']),

    # Diagnostics
    'worst_miss_date':        str(worst_date.date()),
    'worst_miss_error':       float(worst_err),
    'worst_miss_regime':      str(worst_regime),
    'top_feature':            top_feature,

    # Cost
    'wf_total_time_s':        round(wf_total, 1),
    'wf_lstm_time_s':         round(wf_total_lstm, 1),
    'wf_mlp_time_s':          round(wf_total_mlp, 1),
    'avg_epochs_lstm':        round(float(np.mean(epochs_used_lstm)), 1),
    'avg_epochs_mlp':         round(float(np.mean(epochs_used_mlp)), 1),
    'lstm_inference_ms':      round(lstm_inference_ms, 2),
    'mlp_inference_ms':       round(mlp_inference_ms, 2),
    'tf_version':             tf.__version__,
    'garch_persistence_used': GARCH_PERSISTENCE,
}

metrics_path.write_text(json.dumps(metrics, indent=2))

print(f'\nExported notebook_06 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_06'].items():
    print(f'  {k}: {v}')
```

    Predictions saved to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\nb06_predictions.parquet
    

    
    Exported notebook_06 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      primary_seed: 42
      seed_list: [42, 43, 44]
      lstm_units: 32
      lstm_total_params: 5537
      mlp_hidden: 64
      mlp_total_params: 13569
      garch_n_params: 6
      n_features: 10
      lookback: 21
      wf_rmse_lstm: 0.005387486802263018
      wf_mae_lstm: 0.003935097751656807
      wf_rmse_mlp: 0.007588285205721072
      wf_mae_mlp: 0.005570047888646924
      wf_rmse_persistence: 0.0075617841947451326
      wf_mae_persistence: 0.005685889011073224
      lstm_rmse_by_seed: {'42': 0.005387486802263018, '43': 0.005473010436254052, '44': 0.005416717460760347}
      mlp_rmse_by_seed: {'42': 0.007588285205721072, '43': 0.008222599136181208, '44': 0.007546988889905834}
      lstm_rmse_mean: 0.005425738233092473
      lstm_rmse_sd: 4.3469572609575356e-05
      lstm_rmse_min: 0.005387486802263018
      lstm_rmse_max: 0.005473010436254052
      lstm_spread_pct: 1.587449531293709
      mlp_spread_pct: 8.952050362483085
      seeds_beating_garch: 0
      seeds_beating_persistence: 3
      lstm_vs_pers_rmse_pct: 28.753761499740854
      lstm_vs_garch_rmse_pct: -2.884308557530657
      mlp_vs_pers_rmse_pct: -0.3504597631119376
      mlp_vs_garch_rmse_pct: -44.91274042657732
      rmse_material_threshold: 0.01
      lstm_garch_tie: False
      garch_wins: True
      lstm_beats_mlp: True
      dm_stat_lstm_pers: 5.735880153460949
      dm_pval_lstm_pers: 9.70072416991442e-09
      dm_stat_mlp_pers: -0.08011661131088324
      dm_pval_mlp_pers: 0.9361445112999679
      dm_stat_lstm_mlp: 8.33857916870465
      dm_pval_lstm_mlp: 7.518848312675615e-17
      rmse_ci_lstm: [0.004577207165545294, 0.006205798831317348]
      rmse_ci_mlp: [0.006613106877508571, 0.008673246156082576]
      rmse_ci_persistence: [0.006327485030820103, 0.00886358272659801]
      n_bootstrap: 2000
      bootstrap_block: 21
      lb_pval_lstm: 0.977969624348594
      lb_pval_mlp: 1.2721142885968586e-13
      calib_slope_lstm: 0.7239827088552796
      calib_slope_mlp: 0.042889619857762934
      worst_miss_date: 2026-06-05
      worst_miss_error: 0.02281653676036941
      worst_miss_regime: Normal
      top_feature: log_returns
      wf_total_time_s: 2574.9
      wf_lstm_time_s: 684.0
      wf_mlp_time_s: 236.5
      avg_epochs_lstm: 42.2
      avg_epochs_mlp: 36.8
      lstm_inference_ms: 156.83
      mlp_inference_ms: 93.23
      tf_version: 2.21.0
      garch_persistence_used: 0.978945261996311
    
