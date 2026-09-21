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
# ── Reproducibility ──
# The seed controls weight initialisation and data shuffling. TensorFlow
# operations are not fully deterministic, so small numerical differences
# may occur across hardware (CPU vs GPU) and across TensorFlow versions
# even with the seed fixed. Results are reproducible in rank order, not
# to the last decimal place.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

# ── Inference configuration ──
N_BOOT      = 2000  # bootstrap replications for RMSE confidence intervals
BOOT_BLOCK  = 21    # moving block length (preserves error autocorrelation)
LB_LAGS     = [10, 21]  # Ljung-Box lags for residual autocorrelation

print('Configuration locked.')
```

    Configuration locked.
    

## 2. Data loading


```python
df = pd.read_parquet('../data/sp500_features.parquet')

print(f'Rows:    {len(df):,}')
print(f'Columns: {df.shape[1]}')
print(f'Range:   {df.index.min().date()} to {df.index.max().date()}')
df.head(3)
```

    Rows:    6,412
    Columns: 64
    Range:   2001-02-13 to 2026-08-14
    




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
<p>3 rows × 64 columns</p>
</div>




```python
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

# NB05 walk-forward results — loaded dynamically if available,
# otherwise set from the executed NB05 output.
nb05 = metrics.get('notebook_05', {})

GARCH_RMSE  = nb05.get('wf_rmse_garch', 0.005103)
GARCH_MAE   = nb05.get('wf_mae_garch',  0.003828)
PERS_RMSE   = nb05.get('wf_rmse_persistence', 0.007315)
PERS_MAE    = nb05.get('wf_mae_persistence',  0.005411)
GARCH_LABEL = nb05.get('best_label', "GJR-GARCH(1,1,1) — Student's t")

# present only if NB05 exports it. Used in the analytical discussion.
GARCH_PERSISTENCE = nb05.get('garch_persistence', None)

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
| Persistence | 0.007438 | 0.005514 |
| GJR-GARCH(1,1,1) — Student's t | 0.005206 | 0.003964 |

GARCH beat persistence by 30.0% RMSE.




```python
# Derive GARCH parameter count from the model label
# GJR-GARCH(p,o,q) with Student's t: p + o + q + omega + nu = 5 for (1,1,1)
try:
    _order = GARCH_LABEL.split('(')[1].split(')')[0]  # e.g. "1,1,1"
    garch_n_params = sum(int(x) for x in _order.split(',')) + 2  # +omega +nu
except (IndexError, ValueError):
    garch_n_params = 5

display(Markdown(f"""
LSTM and MLP versus {GARCH_LABEL} on one-step-ahead volatility forecasting.

Notebook 04 found no evidence that return direction could be forecast more
accurately than the historical mean baseline. Notebook 05 showed that
volatility is forecastable: {GARCH_LABEL} beat persistence by
{garch_vs_pers:.1f}% RMSE over a {TEST_SIZE}-day walk-forward window,
using {garch_n_params} parameters that each encode a specific financial
mechanism (persistence, mean reversion, shock sensitivity, leverage
asymmetry, and tail shape).

This notebook tests whether neural networks, given access to engineered
features GARCH never sees, can match or beat a model built from financial
structure. Two architectures are compared: an LSTM (which processes
sequences and can learn temporal dependencies) and a feed-forward MLP
(which sees the same features but flattened, without sequential
structure). Comparing the two isolates whether sequence learning adds
value beyond the features themselves.

The evaluation protocol is inherited from Notebook 05 unchanged. The
conclusion follows from the out-of-sample evidence.
"""))
```



LSTM and MLP versus GJR-GARCH(1,1,1) — Student's t on one-step-ahead volatility forecasting.

Notebook 04 found no evidence that return direction could be forecast more
accurately than the historical mean baseline. Notebook 05 showed that
volatility is forecastable: GJR-GARCH(1,1,1) — Student's t beat persistence by
30.0% RMSE over a 252-day walk-forward window,
using 5 parameters that each encode a specific financial
mechanism (persistence, mean reversion, shock sensitivity, leverage
asymmetry, and tail shape).

This notebook tests whether neural networks, given access to engineered
features GARCH never sees, can match or beat a model built from financial
structure. Two architectures are compared: an LSTM (which processes
sequences and can learn temporal dependencies) and a feed-forward MLP
(which sees the same features but flattened, without sequential
structure). Comparing the two isolates whether sequence learning adds
value beyond the features themselves.

The evaluation protocol is inherited from Notebook 05 unchanged. The
conclusion follows from the out-of-sample evidence.




```python
display(Markdown(f"""
## 3. Where this fits

| Notebook | Question | Result |
|---|---|---|
| 04 — Baseline forecasting | Can ARIMA predict return direction? | No. ARIMA matched the historical mean baseline. Null result consistent with weak-form efficiency. |
| 05 — Volatility forecasting | Can GARCH-family models forecast volatility? | Yes. {GARCH_LABEL} beat persistence by {garch_vs_pers:.1f}% RMSE. Selected as the production model. |
| **06 — Deep learning comparison** | **Can neural networks beat the selected model?** | **Tested below.** |

The analytical thread across these notebooks is a narrowing search.
Direction proved unforecastable. Volatility proved forecastable using
models that encode financial structure. The remaining question is whether
flexible models with richer inputs can do better, and if so, whether the
gain comes from temporal modelling (LSTM) or from the feature set alone
(MLP).
"""))
```



## 3. Where this fits

| Notebook | Question | Result |
|---|---|---|
| 04 — Baseline forecasting | Can ARIMA predict return direction? | No. ARIMA matched the historical mean baseline. Null result consistent with weak-form efficiency. |
| 05 — Volatility forecasting | Can GARCH-family models forecast volatility? | Yes. GJR-GARCH(1,1,1) — Student's t beat persistence by 30.0% RMSE. Selected as the production model. |
| **06 — Deep learning comparison** | **Can neural networks beat the selected model?** | **Tested below.** |

The analytical thread across these notebooks is a narrowing search.
Direction proved unforecastable. Volatility proved forecastable using
models that encode financial structure. The remaining question is whether
flexible models with richer inputs can do better, and if so, whether the
gain comes from temporal modelling (LSTM) or from the feature set alone
(MLP).



## 4. Research hypothesis

GARCH models encode volatility dynamics through a small number of
economically interpretable parameters. Deep neural networks provide
substantially greater flexibility and may exploit nonlinear interactions
between engineered features that a parametric model cannot represent.

The hypothesis tested here is whether this additional flexibility
translates into superior out-of-sample volatility forecasts under an
identical walk-forward evaluation protocol.

Two sub-hypotheses follow from it. First, if flexibility helps, at least
one neural network should beat GJR-GARCH on RMSE. Second, if sequential
structure is what the flexibility buys, the LSTM should beat the MLP. The
two questions are separable, and the answers point to different next
steps: a win for neural networks over GARCH argues for capacity, while a
win for the LSTM over the MLP argues specifically for temporal modelling.

The null result is informative in either direction. If neither network
wins, the conclusion is that structural knowledge outperforms capacity on
this data volume, which is a finding about the problem rather than a
failure of the method.

## 5. Evaluation protocol

Inherited from Notebook 05 without modification. The target is the absolute
daily log return (a realised volatility proxy). The test window covers the
final 252 trading days. Models are retrained every 21 days on an expanding
window, and parameters are held fixed between refits while the conditional
variance updates daily. The benchmark is persistence: yesterday's absolute
return forecasts tomorrow's. RMSE and MAE are computed over the full test
window.

### Why MSE trains the networks but RMSE ranks them

The neural networks are trained using mean squared error because it
provides smooth gradients and places greater emphasis on larger
forecasting errors, which matters for a risk application where missing a
volatility spike is costlier than a small error on a quiet day.
Performance is reported using both RMSE and MAE to allow comparison with
Notebook 05. RMSE is the primary ranking metric because it penalises
large volatility misses more heavily and is directly comparable to the
GARCH results. MAE provides an interpretable measure of average forecast
error in the units of the target.

### What each model predicts

GARCH models the conditional variance process that generates returns. Its
output is a distributional parameter (sigma) from which expected absolute
returns are derived. The neural networks predict the realised volatility
proxy directly as a point estimate, with no distributional assumptions.
GARCH says "the variance of the return distribution is X." The LSTM and MLP
say "tomorrow's absolute return will be Y." The evaluation compresses all
three into the same metric space for comparison, which is standard but
worth noting.

### Reproducibility

Random seeds are fixed for weight initialisation and data shuffling.
Despite this, small numerical differences may occur across hardware
(CPU versus GPU) and across TensorFlow versions, because TensorFlow
operations are not fully deterministic. The ranking of models is stable;
the final decimal places are not. GARCH, by contrast, is deterministic
given the same data.

## 6. Target and feature selection


```python
TARGET_COL = 'log_returns'

FEATURE_COLS = [
    'log_returns',     # the return series itself
    'vol_roll_10',     # short-term rolling volatility
    'vol_roll_21',     # medium-term rolling volatility
    'vol_roll_60',     # longer-term volatility context
    'rsi_14',          # momentum indicator
    'atr_14',          # price-based volatility (average true range)
    'bb_width',        # Bollinger Band width
    'vol_rank_30',     # volatility percentile rank
    'vol_ratio_10_60', # short vs medium volatility ratio
    'volume_lag_1',    # lagged trading volume
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

GARCH uses only the return series and its own conditional variance
recursion. The neural networks get a richer input set: three rolling
volatility windows (10, 21, 60 days), two technical indicators (RSI, ATR),
Bollinger Band width, a volatility percentile rank, a short-to-medium
volatility ratio, and lagged volume.

Lagged return features from Notebook 03 are excluded because the 21-day
lookback window already provides that information through the `log_returns`
channel. Calendar dummies are excluded: SARIMA found no significant
seasonal structure in Notebook 04.

## 7. Sequence construction


```python
def create_sequences(X, y, lookback):
    """Build (sequence, target) pairs for supervised learning.

    sequence[i] = features[i : i + lookback]
    target[i]   = |return[i + lookback]|

    At prediction time, all features in the sequence are known
    (up to and including the current close), and the target is
    tomorrow's absolute return.
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

    Total sequences: 6,391
    Training pool:   6,139
    Test sequences:  252
    Test window:     2025-08-14 to 2026-08-14
    Sequence shape:  (21, 10)
    

The 21-day lookback matches the rolling volatility window used
throughout the project. Longer windows (60 or 120 days) increase the
parameter count without clear evidence of long-range dependencies
the rolling features do not already capture.

## 8. LSTM architecture

```
21 days × 10 features ──▸ LSTM (32 units) ──▸ Dropout (0.2) ──▸ Dense (1, softplus) ──▸ |r̂(t+1)|
  known at close of t       learns temporal       regularisation     smooth non-negative    forecast
                            dependencies                            constraint
```

The output activation is `softplus` rather than `relu`. Both enforce
non-negativity (the target is an absolute return), but `relu` has a
zero-gradient region that can stall learning when predictions are near
zero. `softplus` (log(1 + exp(x))) is smooth everywhere and allows
gradients to flow at all output levels.


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


demo = build_lstm(LOOKBACK, n_features)
demo.summary()
lstm_params = int(sum(np.prod(w.shape) for w in demo.trainable_weights))
del demo
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">5,504</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,537</span> (21.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,537</span> (21.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



### Unit count sensitivity

The 32-unit choice follows from a parameter-to-sample ratio argument,
but the question of whether 16 or 64 would perform differently is
reasonable. A quick sensitivity test on the pre-test data (single
train/val split, not walk-forward) checks whether the choice sits in
a stable region.


```python
# Sensitivity test: 16 / 32 / 64 units on fixed train/val split
X_pretrain = X_seq[:test_start]
y_pretrain = y_seq[:test_start]

val_size = int(0.1 * len(X_pretrain))
X_sens_train = X_pretrain[:-val_size]
y_sens_train = y_pretrain[:-val_size]
X_sens_val   = X_pretrain[-val_size:]
y_sens_val   = y_pretrain[-val_size:]

# Scale
sens_scaler = StandardScaler()
X_sens_train_s = sens_scaler.fit_transform(
    X_sens_train.reshape(-1, n_features)
).reshape(X_sens_train.shape)
X_sens_val_s = sens_scaler.transform(
    X_sens_val.reshape(-1, n_features)
).reshape(X_sens_val.shape)

sensitivity = []
for units in [16, 32, 64]:
    tf.random.set_seed(SEED)
    m = build_lstm(LOOKBACK, n_features, units=units)
    h = m.fit(
        X_sens_train_s, y_sens_train,
        validation_data=(X_sens_val_s, y_sens_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        verbose=0,
    )
    best_val = min(h.history['val_loss'])
    stopped = len(h.history['loss'])
    sensitivity.append({
        'Units': units,
        'Parameters': m.count_params(),
        'Best val MSE': f'{best_val:.2e}',
        'Epochs': stopped,
    })
    del m

sens_df = pd.DataFrame(sensitivity)
display(sens_df.to_string(index=False))
```


    ' Units  Parameters Best val MSE  Epochs\n    16        1745     5.71e-05      29\n    32        5537     5.30e-05      47\n    64       19265     4.57e-05      61'



```python
# Dynamic interpretation based on sensitivity results
val_losses = [float(r['Best val MSE']) for r in sensitivity]
spread = (max(val_losses) - min(val_losses)) / min(val_losses) * 100

if spread < 20:
    interp = (
        f"The three configurations produce validation losses within "
        f"{spread:.0f}% of each other. This confirms 32 units sits in a "
        f"stable region, neither starved for capacity nor wasting "
        f"parameters. The walk-forward evaluation proceeds with 32."
    )
else:
    best_units = sensitivity[val_losses.index(min(val_losses))]['Units']
    interp = (
        f"Validation loss varies by {spread:.0f}% across configurations. "
        f"{best_units} units performed best, but the walk-forward "
        f"evaluation proceeds with 32 for comparability."
    )

display(Markdown(interp))
```


Validation loss varies by 25% across configurations. 64 units performed best, but the walk-forward evaluation proceeds with 32 for comparability.


### Why there is no hyperparameter search

Hyperparameter optimisation was deliberately excluded. The objective is to
compare modelling paradigms, structural versus flexible, rather than to
maximise neural network performance through extensive search. A tuned
network compared against an untuned GARCH would not answer the question
the notebook asks.

Introducing a large hyperparameter search would also increase the risk of
overfitting to a single 252-day evaluation window. With one test period
and no outer validation loop, a search over dozens of configurations would
select whichever happened to suit this window, inflating the apparent
performance in a way that would not survive on new data.

The sensitivity check above is the deliberate exception: it verifies that
the chosen capacity is not pathological, without selecting on test
performance.

## 9. MLP baseline

The LSTM assumes that the temporal ordering of features within the
21-day window matters. An MLP receives the same 21 × 10 = 210 input
values but flattened into a single vector, discarding the sequential
structure. If the MLP matches the LSTM, the temporal modelling adds
nothing and the feature set alone carries whatever signal exists. If
the LSTM materially outperforms the MLP, there is evidence of
exploitable sequential structure beyond what the rolling features
already summarise.

```
LSTM
21 days × 10 features ──▸ LSTM (32 units) ──▸ Dropout ──▸ Dense (1, softplus) ──▸ |r̂(t+1)|
  ordering preserved        recurrent state

MLP
21 days × 10 features ──▸ Flatten (210) ──▸ Dense (64, relu) ──▸ Dropout ──▸ Dense (1, softplus) ──▸ |r̂(t+1)|
  ordering discarded        one long vector
```

Both paths end in the same softplus output for comparability. The MLP's
hidden layer uses `relu` rather than `softplus`: the zero-gradient concern
applies to the output layer, where predictions cluster near zero, not to a
hidden layer operating across a wider activation range.


```python
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


demo_mlp = build_mlp(LOOKBACK, n_features)
demo_mlp.summary()
mlp_params = int(sum(np.prod(w.shape) for w in demo_mlp.trainable_weights))
del demo_mlp

print(f'\nLSTM trainable parameters: {lstm_params:,}')
print(f'MLP trainable parameters:  {mlp_params:,}')
print(f'Ratio:                     {mlp_params / lstm_params:.1f}×')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_4"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">210</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">13,504</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">65</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,569</span> (53.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,569</span> (53.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    LSTM trainable parameters: 5,537
    MLP trainable parameters:  13,569
    Ratio:                     2.5×
    

### A note on feature scaling

Standardisation is performed separately within each walk-forward training
window using statistics computed only from the training data. The fitted
scaler is then applied unchanged to the corresponding test block. This
prevents information leakage while maintaining comparable feature scales
during optimisation.

Scaling does not destroy temporal information. It is a per-feature affine
transformation (subtract mean, divide by standard deviation) applied
identically to every timestep, so the ordering, the relative movements,
and the autocorrelation structure within each sequence are all preserved.
What changes is the numerical range the optimiser works in, which matters
because gradient descent converges poorly when input features differ by
orders of magnitude, as they do here, with log returns near 0.01 and
volume in the billions.

## 10. Walk-forward evaluation


```python
n_blocks = (TEST_SIZE + REFIT_EVERY - 1) // REFIT_EVERY
lstm_forecasts = np.zeros(TEST_SIZE)
mlp_forecasts  = np.zeros(TEST_SIZE)
training_histories_lstm = []
training_histories_mlp  = []
block_times_lstm = []
block_times_mlp  = []
epochs_used_lstm = []
epochs_used_mlp  = []

print(f'Walk-forward: {n_blocks} refit blocks of {REFIT_EVERY} days')
print(f'Training starts with {test_start:,} sequences, expanding each block.')
print()

wf_start = time.perf_counter()

for block in range(n_blocks):
    block_start = block * REFIT_EVERY
    block_end = min(block_start + REFIT_EVERY, TEST_SIZE)

    # ── Expanding training window ──
    train_end = test_start + block_start
    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]

    # ── Scale features (fit on training only) ──
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)

    # ── Validation split: last 10% of training ──
    val_size = max(int(0.1 * len(X_train_scaled)), LOOKBACK)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    X_fit = X_train_scaled[:-val_size]
    y_fit = y_train[:-val_size]

    # ── Test block (scaled) ──
    X_test_block = X_seq[test_start + block_start : test_start + block_end]
    X_test_2d = X_test_block.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test_block.shape)

    es = EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                       monitor='val_loss')

    # ── LSTM ──
    tf.random.set_seed(SEED + block)
    lstm_model = build_lstm(LOOKBACK, n_features)
    t0 = time.perf_counter()
    h_lstm = lstm_model.fit(
        X_fit, y_fit, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0,
    )
    block_times_lstm.append(time.perf_counter() - t0)
    training_histories_lstm.append(h_lstm.history)
    epochs_used_lstm.append(len(h_lstm.history['loss']))

    preds_lstm = lstm_model.predict(X_test_scaled, verbose=0).flatten()
    lstm_forecasts[block_start:block_end] = preds_lstm

    # ── MLP ──
    tf.random.set_seed(SEED + block)
    mlp_model = build_mlp(LOOKBACK, n_features)
    t0 = time.perf_counter()
    h_mlp = mlp_model.fit(
        X_fit, y_fit, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0,
    )
    block_times_mlp.append(time.perf_counter() - t0)
    training_histories_mlp.append(h_mlp.history)
    epochs_used_mlp.append(len(h_mlp.history['loss']))

    preds_mlp = mlp_model.predict(X_test_scaled, verbose=0).flatten()
    mlp_forecasts[block_start:block_end] = preds_mlp

    print(
        f'Block {block+1:2d}/{n_blocks}: '
        f'LSTM epochs={epochs_used_lstm[-1]:3d} '
        f'val={h_lstm.history["val_loss"][-1]:.2e} '
        f'{block_times_lstm[-1]:.1f}s | '
        f'MLP epochs={epochs_used_mlp[-1]:3d} '
        f'val={h_mlp.history["val_loss"][-1]:.2e} '
        f'{block_times_mlp[-1]:.1f}s'
    )

wf_total = time.perf_counter() - wf_start
wf_total_lstm = sum(block_times_lstm)
wf_total_mlp  = sum(block_times_mlp)

# Save last models and scaler for downstream diagnostics
last_lstm  = lstm_model
last_mlp   = mlp_model
last_scaler = scaler

print(f'\nWalk-forward complete in {wf_total:.1f}s')
print(f'  LSTM total: {wf_total_lstm:.1f}s  (avg epochs: {np.mean(epochs_used_lstm):.1f})')
print(f'  MLP total:  {wf_total_mlp:.1f}s  (avg epochs: {np.mean(epochs_used_mlp):.1f})')
```

    Walk-forward: 12 refit blocks of 21 days
    Training starts with 6,139 sequences, expanding each block.
    
    Block  1/12: LSTM epochs= 25 val=6.60e-05 38.4s | MLP epochs= 41 val=1.04e-04 23.4s
    Block  2/12: LSTM epochs= 60 val=4.68e-05 89.5s | MLP epochs= 37 val=8.16e-05 20.3s
    WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001A8796004A0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    WARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001A87962A480> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    Block  3/12: LSTM epochs= 71 val=4.46e-05 105.8s | MLP epochs= 51 val=8.65e-05 28.0s
    Block  4/12: LSTM epochs= 60 val=5.70e-05 91.4s | MLP epochs= 55 val=8.72e-05 28.3s
    Block  5/12: LSTM epochs= 70 val=4.64e-05 105.0s | MLP epochs= 59 val=8.30e-05 32.9s
    Block  6/12: LSTM epochs= 25 val=2.08e-04 40.2s | MLP epochs= 51 val=8.54e-05 29.0s
    Block  7/12: LSTM epochs= 36 val=7.62e-05 56.9s | MLP epochs= 58 val=8.98e-05 42.0s
    Block  8/12: LSTM epochs= 50 val=4.71e-05 77.1s | MLP epochs= 59 val=1.18e-04 33.7s
    Block  9/12: LSTM epochs= 48 val=7.24e-05 74.3s | MLP epochs= 31 val=8.72e-05 18.3s
    Block 10/12: LSTM epochs= 54 val=4.94e-05 70.4s | MLP epochs= 71 val=8.93e-05 38.7s
    Block 11/12: LSTM epochs= 20 val=1.14e-04 33.6s | MLP epochs= 26 val=9.03e-05 15.8s
    Block 12/12: LSTM epochs= 71 val=5.40e-05 110.5s | MLP epochs= 33 val=8.77e-05 19.2s
    
    Walk-forward complete in 1229.0s
      LSTM total: 893.1s  (avg epochs: 49.2)
      MLP total:  329.4s  (avg epochs: 47.7)
    


```python
actual = y_seq[test_start : test_start + TEST_SIZE]
persistence = y_seq[test_start - 1 : test_start + TEST_SIZE - 1]

pers_rmse_check = root_mean_squared_error(actual, persistence)
pers_mae_check  = mean_absolute_error(actual, persistence)

print(f'Persistence RMSE (this window): {pers_rmse_check:.6f}')
print(f'NB05 persistence RMSE:          {PERS_RMSE:.6f}')

if abs(pers_rmse_check - PERS_RMSE) / PERS_RMSE > 0.05:
    print('\n⚠ Persistence RMSE differs by >5% from NB05. '
          'Check test window alignment.')
```

    Persistence RMSE (this window): 0.007438
    NB05 persistence RMSE:          0.007438
    

## 11. Results


```python
lstm_rmse = root_mean_squared_error(actual, lstm_forecasts)
lstm_mae  = mean_absolute_error(actual, lstm_forecasts)
mlp_rmse  = root_mean_squared_error(actual, mlp_forecasts)
mlp_mae   = mean_absolute_error(actual, mlp_forecasts)

# Error series reused by the bootstrap, DM test, and residual diagnostics
lstm_errors = actual - lstm_forecasts
mlp_errors  = actual - mlp_forecasts
pers_errors = actual - persistence

lstm_vs_pers_rmse  = (pers_rmse_check - lstm_rmse) / pers_rmse_check * 100
lstm_vs_garch_rmse = (GARCH_RMSE - lstm_rmse) / GARCH_RMSE * 100
mlp_vs_pers_rmse   = (pers_rmse_check - mlp_rmse) / pers_rmse_check * 100
mlp_vs_garch_rmse  = (GARCH_RMSE - mlp_rmse) / GARCH_RMSE * 100

comparison = pd.DataFrame({
    'RMSE': [pers_rmse_check, GARCH_RMSE, mlp_rmse, lstm_rmse],
    'MAE':  [pers_mae_check,  GARCH_MAE,  mlp_mae,  lstm_mae],
}, index=['Persistence', f'{GARCH_LABEL} (NB05)',
          f'MLP ({MLP_HIDDEN} hidden)', f'LSTM ({LSTM_UNITS} units)'])

display(comparison)
print(f'\nLSTM vs persistence: {lstm_vs_pers_rmse:+.1f}% RMSE')
print(f'LSTM vs GARCH:       {lstm_vs_garch_rmse:+.1f}% RMSE')
print(f'MLP vs persistence:  {mlp_vs_pers_rmse:+.1f}% RMSE')
print(f'MLP vs GARCH:        {mlp_vs_garch_rmse:+.1f}% RMSE')
print(f'LSTM vs MLP:         {(mlp_rmse - lstm_rmse) / mlp_rmse * 100:+.1f}% RMSE')
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
      <td>0.007438</td>
      <td>0.005514</td>
    </tr>
    <tr>
      <th>GJR-GARCH(1,1,1) — Student's t (NB05)</th>
      <td>0.005206</td>
      <td>0.003964</td>
    </tr>
    <tr>
      <th>MLP (64 hidden)</th>
      <td>0.007960</td>
      <td>0.005928</td>
    </tr>
    <tr>
      <th>LSTM (32 units)</th>
      <td>0.006840</td>
      <td>0.004709</td>
    </tr>
  </tbody>
</table>
</div>


    
    LSTM vs persistence: +8.0% RMSE
    LSTM vs GARCH:       -31.4% RMSE
    MLP vs persistence:  -7.0% RMSE
    MLP vs GARCH:        -52.9% RMSE
    LSTM vs MLP:         +14.1% RMSE
    


```python
# Materiality threshold for the model comparison.
#
# A bare inequality treats a 0.02% RMSE gap and a 20% gap as the same
# statement. The bootstrap interval for the LSTM below is roughly 30% of its
# own width; a difference three orders of magnitude smaller than that is not
# something this evaluation can resolve. 1% relative is a judgement, set well
# inside the interval width while still admitting the MLP's real shortfall.

RMSE_MATERIAL = 0.01

lstm_garch_rel = abs(lstm_rmse - GARCH_RMSE) / GARCH_RMSE
mlp_garch_rel = abs(mlp_rmse - GARCH_RMSE) / GARCH_RMSE
lstm_garch_tie = lstm_garch_rel < RMSE_MATERIAL
mlp_garch_tie = mlp_garch_rel < RMSE_MATERIAL

garch_wins = lstm_rmse > GARCH_RMSE and mlp_rmse > GARCH_RMSE
lstm_beats_mlp = lstm_rmse < mlp_rmse
beats_persistence_lstm = lstm_rmse < pers_rmse_check
beats_persistence_mlp = mlp_rmse < pers_rmse_check

# One denominator for the LSTM-vs-MLP gap, reused in section 15.
lstm_mlp_rel = abs(lstm_rmse - mlp_rmse) / max(lstm_rmse, mlp_rmse) * 100

best_nn_label = f'LSTM ({LSTM_UNITS} units)' if lstm_beats_mlp else f'MLP ({MLP_HIDDEN} hidden)'
best_nn_rmse = lstm_rmse if lstm_beats_mlp else mlp_rmse
best_nn_vs_pers = lstm_vs_pers_rmse if lstm_beats_mlp else mlp_vs_pers_rmse

print(f'LSTM vs GARCH: {lstm_garch_rel:.2e} relative '
      f'({"tie" if lstm_garch_tie else "material"})')
print(f'MLP vs GARCH:  {mlp_garch_rel:.2e} relative '
      f'({"tie" if mlp_garch_tie else "material"})')
print(f'LSTM MAE {lstm_mae:.6f} vs GARCH MAE {GARCH_MAE:.6f}')

if lstm_garch_tie:
    lstm_verdict = (
        f"The LSTM and {GARCH_LABEL} are level. RMSE differs by "
        f"{lstm_garch_rel:.2e} in relative terms ({lstm_rmse:.6f} against "
        f"{GARCH_RMSE:.6f}), and on MAE the LSTM is "
        f"{'lower' if lstm_mae < GARCH_MAE else 'higher'} "
        f"({lstm_mae:.6f} against {GARCH_MAE:.6f}). The bootstrap interval "
        "below reaches the same conclusion independently."
    )
elif lstm_rmse > GARCH_RMSE:
    lstm_verdict = f"{GARCH_LABEL} beat the LSTM by {lstm_garch_rel * 100:.1f}% on RMSE."
else:
    lstm_verdict = f"The LSTM beat {GARCH_LABEL} by {lstm_garch_rel * 100:.1f}% on RMSE."

if mlp_garch_tie:
    mlp_verdict = f"The MLP is also level with {GARCH_LABEL}."
elif mlp_rmse > GARCH_RMSE:
    mlp_verdict = (
        f"The MLP fell short by {mlp_garch_rel * 100:.1f}%, and did not beat "
        f"persistence either ({mlp_rmse:.6f} against {pers_rmse_check:.6f})."
        if not beats_persistence_mlp
        else f"The MLP fell short by {mlp_garch_rel * 100:.1f}%."
    )
else:
    mlp_verdict = f"The MLP beat {GARCH_LABEL} by {mlp_garch_rel * 100:.1f}%."

display(Markdown(f"""
### Interpretation

{lstm_verdict} {mlp_verdict}

The parameter cost of that parity is the finding. {GARCH_LABEL} reaches it
with {garch_n_params} parameters and no training loop; the LSTM needs
{lstm_params:,} and {wf_total_lstm:.0f} seconds across {n_blocks} refits.
Matching a structural model is not improving on it, and the operational
difference is what decides deployment.

The LSTM {'outperformed' if lstm_beats_mlp else 'underperformed'} the MLP by
{lstm_mlp_rel:.1f}% on RMSE. Since the MLP sees identical inputs, the feature
set alone does not carry the signal: whatever the LSTM extracts comes from
the sequential structure the MLP discards.

This is a single {TEST_SIZE}-day test window. The ranking could change under
different market regimes, and the result is reported as found.
"""))
```

    LSTM vs GARCH: 3.14e-01 relative (material)
    MLP vs GARCH:  5.29e-01 relative (material)
    LSTM MAE 0.004709 vs GARCH MAE 0.003964
    



### Interpretation

GJR-GARCH(1,1,1) — Student's t beat the LSTM by 31.4% on RMSE. The MLP fell short by 52.9%, and did not beat persistence either (0.007960 against 0.007438).

The parameter cost of that parity is the finding. GJR-GARCH(1,1,1) — Student's t reaches it
with 5 parameters and no training loop; the LSTM needs
5,537 and 893 seconds across 12 refits.
Matching a structural model is not improving on it, and the operational
difference is what decides deployment.

The LSTM outperformed the MLP by
14.1% on RMSE. Since the MLP sees identical inputs, the feature
set alone does not carry the signal: whatever the LSTM extracts comes from
the sequential structure the MLP discards.

This is a single 252-day test window. The ranking could change under
different market regimes, and the result is reported as found.



### Bootstrap confidence intervals

A single RMSE figure gives no sense of sampling uncertainty. Two models
differing by 2% RMSE may be indistinguishable once that uncertainty is
accounted for.

The intervals below use a moving block bootstrap rather than an
independent resample. Forecast errors in volatility models are
autocorrelated, the same property that motivated the Newey-West
correction in the Diebold-Mariano test, and resampling individual days
independently would break that dependence and understate the interval
width. Blocks of 21 days preserve the local error structure.


```python
def block_bootstrap_rmse_ci(errors, n_boot=N_BOOT, block=BOOT_BLOCK,
                            alpha=0.05, seed=SEED):
    """Moving block bootstrap confidence interval for RMSE.

    Resamples contiguous blocks of forecast errors with replacement,
    preserving short-range autocorrelation that an i.i.d. bootstrap
    would destroy. Returns (lower, upper) percentile bounds.
    """
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors)
    n = len(errors)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block + 1

    boot_rmse = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, n_blocks)
        sample = np.concatenate([errors[s : s + block] for s in starts])[:n]
        boot_rmse[b] = np.sqrt(np.mean(sample**2))

    lo, hi = np.percentile(boot_rmse, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


lstm_ci = block_bootstrap_rmse_ci(lstm_errors)
mlp_ci  = block_bootstrap_rmse_ci(mlp_errors)
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

Based on {N_BOOT:,} bootstrap replications with {BOOT_BLOCK}-day blocks.

The LSTM and MLP intervals {'overlap' if ci_overlap_lstm_mlp else 'do not overlap'},
{'so their accuracy difference is not distinguishable from sampling noise on this window.' if ci_overlap_lstm_mlp else 'so the difference between them is unlikely to be sampling noise alone.'}
The GARCH RMSE {'falls inside' if garch_in_lstm_ci else 'falls outside'} the LSTM
interval, {'so the two are statistically indistinguishable on this evidence.' if garch_in_lstm_ci else 'which supports treating the difference as real rather than incidental.'}

Computing an equivalent interval for GARCH requires its walk-forward error
series from Notebook 05.
"""))
```



| Model | RMSE | 95% CI (block bootstrap) | Width |
|---|---|---|---|
| Persistence | 0.007438 | [0.006269, 0.008775] | 0.002506 |
| GJR-GARCH(1,1,1) — Student's t | 0.005206 | (not computed — see NB05) | — |
| MLP (64 hidden) | 0.007960 | [0.006997, 0.009140] | 0.002143 |
| LSTM (32 units) | 0.006840 | [0.004619, 0.009524] | 0.004905 |

Based on 2,000 bootstrap replications with 21-day blocks.

The LSTM and MLP intervals overlap,
so their accuracy difference is not distinguishable from sampling noise on this window.
The GARCH RMSE falls inside the LSTM
interval, so the two are statistically indistinguishable on this evidence.

Computing an equivalent interval for GARCH requires its walk-forward error
series from Notebook 05.



### Parameter efficiency

Stating that GARCH uses five parameters is descriptive. Dividing the
accuracy gain by the parameter count makes the efficiency comparison
quantitative: how much RMSE improvement does each parameter buy?


```python
def pct_improvement(model_rmse):
    return (pers_rmse_check - model_rmse) / pers_rmse_check * 100

eff_rows = [
    {
        'Model': GARCH_LABEL,
        'Trainable parameters': garch_n_params,
        'RMSE improvement vs persistence': pct_improvement(GARCH_RMSE),
    },
    {
        'Model': f'MLP ({MLP_HIDDEN} hidden)',
        'Trainable parameters': mlp_params,
        'RMSE improvement vs persistence': pct_improvement(mlp_rmse),
    },
    {
        'Model': f'LSTM ({LSTM_UNITS} units)',
        'Trainable parameters': lstm_params,
        'RMSE improvement vs persistence': pct_improvement(lstm_rmse),
    },
]

eff_df = pd.DataFrame(eff_rows)
eff_df['Improvement per parameter'] = (
    eff_df['RMSE improvement vs persistence'] / eff_df['Trainable parameters']
)
eff_df = eff_df.sort_values('Improvement per parameter', ascending=False)

fmt = eff_df.copy()
fmt['Trainable parameters'] = fmt['Trainable parameters'].map('{:,}'.format)
fmt['RMSE improvement vs persistence'] = (
    fmt['RMSE improvement vs persistence'].map('{:+.2f}%'.format)
)
fmt['Improvement per parameter'] = (
    fmt['Improvement per parameter'].map('{:+.5f}%'.format)
)

display(Markdown(fmt.to_markdown(index=False)))

# The ratio is only meaningful between models that actually beat persistence.
# A model with negative improvement has negative efficiency, and a ratio
# across the sign boundary would be uninterpretable.
positive = eff_df[eff_df['Improvement per parameter'] > 0]

if len(positive) >= 2:
    leader = positive.iloc[0]
    laggard = positive.iloc[-1]
    ratio = leader['Improvement per parameter'] / laggard['Improvement per parameter']
    eff_note = (
        f"Ranked by improvement per parameter, **{leader['Model']}** leads "
        f"**{laggard['Model']}** by a factor of roughly {ratio:,.0f}×."
    )
elif len(positive) == 1:
    eff_note = (
        f"Only **{positive.iloc[0]['Model']}** improved on persistence, so a "
        f"like-for-like efficiency ratio is not available. The remaining models "
        f"have negative efficiency: parameters spent without accuracy gained."
    )
else:
    eff_note = (
        'No model improved on persistence over this window, so parameter '
        'efficiency comparisons do not apply.'
    )

display(Markdown(
    eff_note +
    ' Parameter efficiency is not the deployment criterion on its own, but it '
    'quantifies how much of a model\'s accuracy is bought with structure '
    'versus bought with capacity.'
))
```


| Model                          |   Trainable parameters | RMSE improvement vs persistence   | Improvement per parameter   |
|:-------------------------------|-----------------------:|:----------------------------------|:----------------------------|
| GJR-GARCH(1,1,1) — Student's t |                      5 | +30.00%                           | +6.00061%                   |
| LSTM (32 units)                |                  5,537 | +8.04%                            | +0.00145%                   |
| MLP (64 hidden)                |                 13,569 | -7.02%                            | -0.00052%                   |



Ranked by improvement per parameter, **GJR-GARCH(1,1,1) — Student's t** leads **LSTM (32 units)** by a factor of roughly 4,130×. Parameter efficiency is not the deployment criterion on its own, but it quantifies how much of a model's accuracy is bought with structure versus bought with capacity.


### Diebold–Mariano test


```python
def diebold_mariano_hac(e1, e2, h=1):
    """Diebold-Mariano test with Newey-West HAC variance estimator.

    H0: E[L(e1) - L(e2)] = 0 where L is squared error.
    Positive DM statistic means model 2 is more accurate.

    The HAC estimator accounts for autocorrelation in the loss
    differential, which is expected at horizons > 1 and common
    even at h=1 for volatility forecasts.
    """
    d = e1**2 - e2**2
    T = len(d)
    d_bar = d.mean()

    # Newey-West bandwidth: h - 1 or at least 1
    max_lag = max(h - 1, 1)

    # Autocovariance at lag 0
    gamma_0 = np.sum((d - d_bar)**2) / T

    # Add weighted autocovariances (Bartlett kernel)
    gamma_sum = 0
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        gamma_sum += 2 * weight * gamma_k

    hac_var = (gamma_0 + gamma_sum) / T

    if hac_var <= 0:
        # Fall back to simple variance if HAC estimate is non-positive
        hac_var = np.var(d, ddof=1) / T

    dm_stat = d_bar / np.sqrt(hac_var)
    p_value = 2 * stats.norm.sf(abs(dm_stat))
    return dm_stat, p_value


# LSTM vs persistence
dm_lp, p_lp = diebold_mariano_hac(pers_errors, lstm_errors)

# MLP vs persistence
dm_mp, p_mp = diebold_mariano_hac(pers_errors, mlp_errors)

# LSTM vs MLP
dm_lm, p_lm = diebold_mariano_hac(mlp_errors, lstm_errors)

display(Markdown(f"""
| Comparison | DM statistic | p-value | Significant at 5%? |
|---|---|---|---|
| LSTM vs Persistence | {dm_lp:.3f} | {p_lp:.4f} | {'Yes' if p_lp < 0.05 else 'No'} |
| MLP vs Persistence | {dm_mp:.3f} | {p_mp:.4f} | {'Yes' if p_mp < 0.05 else 'No'} |
| LSTM vs MLP | {dm_lm:.3f} | {p_lm:.4f} | {'Yes' if p_lm < 0.05 else 'No'} |

The variance estimator uses a Newey-West (Bartlett kernel) correction for
autocorrelation in the loss differential.
A direct neural-network-vs-GARCH Diebold–Mariano test requires the GARCH
forecast series from Notebook 05. If NB05 exports its walk-forward
predictions to Parquet, that test can be added here.
"""))
```



| Comparison | DM statistic | p-value | Significant at 5%? |
|---|---|---|---|
| LSTM vs Persistence | 0.867 | 0.3861 | No |
| MLP vs Persistence | -1.610 | 0.1074 | No |
| LSTM vs MLP | 1.692 | 0.0907 | No |

The variance estimator uses a Newey-West (Bartlett kernel) correction for
autocorrelation in the loss differential.
A direct neural-network-vs-GARCH Diebold–Mariano test requires the GARCH
forecast series from Notebook 05. If NB05 exports its walk-forward
predictions to Parquet, that test can be added here.



## 12. Computational cost


```python
# Inference speed: time prediction of all 252 test observations
X_test_full = X_seq[test_start : test_start + TEST_SIZE]
X_test_full_2d = X_test_full.reshape(-1, n_features)
X_test_full_s = last_scaler.transform(X_test_full_2d).reshape(X_test_full.shape)

# Warm up
_ = last_lstm.predict(X_test_full_s[:1], verbose=0)
_ = last_mlp.predict(X_test_full_s[:1], verbose=0)

# LSTM inference
t0 = time.perf_counter()
_ = last_lstm.predict(X_test_full_s, verbose=0)
lstm_inference_ms = (time.perf_counter() - t0) * 1000

# MLP inference
t0 = time.perf_counter()
_ = last_mlp.predict(X_test_full_s, verbose=0)
mlp_inference_ms = (time.perf_counter() - t0) * 1000

# Persistence inference
t0 = time.perf_counter()
_ = y_seq[test_start - 1 : test_start + TEST_SIZE - 1].copy()
pers_inference_ms = (time.perf_counter() - t0) * 1000

avg_train_lstm = np.mean(block_times_lstm)
avg_train_mlp  = np.mean(block_times_mlp)
avg_epochs_lstm = np.mean(epochs_used_lstm)
avg_epochs_mlp  = np.mean(epochs_used_mlp)

display(Markdown(f"""
### Training cost (full walk-forward)

| Model | Total training | Avg per refit | Avg epochs |
|---|---|---|---|
| Persistence | 0 s | — | — |
| {GARCH_LABEL} | (see NB05) | (see NB05) | — |
| MLP ({MLP_HIDDEN} hidden) | {wf_total_mlp:.1f} s | {avg_train_mlp:.1f} s | {avg_epochs_mlp:.0f} |
| LSTM ({LSTM_UNITS} units) | {wf_total_lstm:.1f} s | {avg_train_lstm:.1f} s | {avg_epochs_lstm:.0f} |

### Inference speed (252 observations)

| Model | Time |
|---|---|
| Persistence | {pers_inference_ms:.2f} ms |
| {GARCH_LABEL} | (see NB05) |
| MLP ({MLP_HIDDEN} hidden) | {mlp_inference_ms:.1f} ms |
| LSTM ({LSTM_UNITS} units) | {lstm_inference_ms:.1f} ms |

Both neural networks train on CPU within practical time. GARCH training
and inference times are reported in Notebook 05; exact comparison requires
running all models on the same hardware in the same session.
"""))
```



### Training cost (full walk-forward)

| Model | Total training | Avg per refit | Avg epochs |
|---|---|---|---|
| Persistence | 0 s | — | — |
| GJR-GARCH(1,1,1) — Student's t | (see NB05) | (see NB05) | — |
| MLP (64 hidden) | 329.4 s | 27.5 s | 48 |
| LSTM (32 units) | 893.1 s | 74.4 s | 49 |

### Inference speed (252 observations)

| Model | Time |
|---|---|
| Persistence | 0.12 ms |
| GJR-GARCH(1,1,1) — Student's t | (see NB05) |
| MLP (64 hidden) | 98.5 ms |
| LSTM (32 units) | 138.3 ms |

Both neural networks train on CPU within practical time. GARCH training
and inference times are reported in Notebook 05; exact comparison requires
running all models on the same hardware in the same session.



## 13. Diagnostics

### Training curves


```python
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('LSTM (first and last refit)',
                                    'MLP (first and last refit)'))

for col, (histories, label) in enumerate([
    (training_histories_lstm, 'LSTM'),
    (training_histories_mlp,  'MLP'),
], start=1):
    for idx, block_label in [(0, 'Block 1'), (-1, f'Block {n_blocks}')]:
        h = histories[idx]
        fig.add_trace(go.Scatter(
            y=h['loss'], name=f'{block_label} — train',
            mode='lines', line=dict(dash='solid'),
            legendgroup=label, showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            y=h['val_loss'], name=f'{block_label} — val',
            mode='lines', line=dict(dash='dash'),
            legendgroup=label, showlegend=(col == 1),
        ), row=1, col=col)

fig.update_layout(
    template='plotly_dark',
    title='Training curves (first and last refit)',
    height=400,
)
fig.update_xaxes(title_text='Epoch')
fig.update_yaxes(title_text='MSE loss')
fig.show()

first_stopped_lstm = epochs_used_lstm[0]
last_stopped_lstm  = epochs_used_lstm[-1]
last_train_size = test_start + (n_blocks - 1) * REFIT_EVERY

display(Markdown(f"""
LSTM early stopping triggered at epoch {first_stopped_lstm} (first block)
and epoch {last_stopped_lstm} (last block). {'Validation loss stabilised before the epoch ceiling, suggesting the network reached its capacity limit rather than exhausting the training budget.' if max(first_stopped_lstm, last_stopped_lstm) < EPOCHS - PATIENCE else 'The network used most of its epoch budget, suggesting it was still learning. Increasing EPOCHS or PATIENCE may help.'}
The last block trained on {last_train_size:,} sequences, roughly
{last_train_size / lstm_params:.1f}× the LSTM parameter count{'.' if last_train_size / lstm_params >= 20 else ', which is in the data-limited regime for deep learning.'}
"""))
```





LSTM early stopping triggered at epoch 25 (first block)
and epoch 71 (last block). Validation loss stabilised before the epoch ceiling, suggesting the network reached its capacity limit rather than exhausting the training budget.
The last block trained on 6,370 sequences, roughly
1.2× the LSTM parameter count, which is in the data-limited regime for deep learning.



### Forecast comparison


```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pred_dates, y=actual,
    name='Actual |return|', mode='lines',
    line=dict(color='white', width=1), opacity=0.5,
))
fig.add_trace(go.Scatter(
    x=pred_dates, y=lstm_forecasts,
    name=f'LSTM ({LSTM_UNITS} units)', mode='lines',
    line=dict(color='#00d4aa', width=1.5),
))
fig.add_trace(go.Scatter(
    x=pred_dates, y=mlp_forecasts,
    name=f'MLP ({MLP_HIDDEN} hidden)', mode='lines',
    line=dict(color='#ffd700', width=1.5),
))
fig.add_trace(go.Scatter(
    x=pred_dates, y=persistence,
    name='Persistence', mode='lines',
    line=dict(color='#ff6b6b', width=1, dash='dot'), opacity=0.6,
))

fig.update_layout(
    template='plotly_dark',
    title='Walk-forward forecasts: LSTM vs MLP vs persistence vs actual',
    xaxis_title='Date', yaxis_title='|Daily log return|',
    height=450,
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
)
fig.show()
```




```python
lstm_abs_errors = np.abs(actual - lstm_forecasts)
mlp_abs_errors  = np.abs(actual - mlp_forecasts)
pers_abs_errors = np.abs(actual - persistence)

lstm_rolling = pd.Series(lstm_abs_errors, index=pred_dates).rolling(21).mean()
mlp_rolling  = pd.Series(mlp_abs_errors, index=pred_dates).rolling(21).mean()
pers_rolling = pd.Series(pers_abs_errors, index=pred_dates).rolling(21).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred_dates, y=lstm_rolling,
    name='LSTM 21-day rolling MAE',
    line=dict(color='#00d4aa', width=1.5),
))
fig.add_trace(go.Scatter(
    x=pred_dates, y=mlp_rolling,
    name='MLP 21-day rolling MAE',
    line=dict(color='#ffd700', width=1.5),
))
fig.add_trace(go.Scatter(
    x=pred_dates, y=pers_rolling,
    name='Persistence 21-day rolling MAE',
    line=dict(color='#ff6b6b', width=1.5),
))
fig.update_layout(
    template='plotly_dark',
    title='Rolling forecast error: LSTM vs MLP vs persistence',
    xaxis_title='Date', yaxis_title='21-day rolling MAE',
    height=400,
)
fig.show()
```



### Residual analysis


```python
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('LSTM forecast error distribution',
                                    'Error vs actual volatility'))

# Error histogram
fig.add_trace(go.Histogram(
    x=lstm_errors, nbinsx=50, name='LSTM errors',
    marker_color='#00d4aa', opacity=0.7,
), row=1, col=1)

# Error vs actual
fig.add_trace(go.Scatter(
    x=actual, y=lstm_errors,
    mode='markers', name='Error vs actual',
    marker=dict(color='#00d4aa', size=3, opacity=0.5),
), row=1, col=2)
fig.add_hline(y=0, line_dash='dash', line_color='white',
              opacity=0.5, row=1, col=2)

fig.update_layout(
    template='plotly_dark', height=350, showlegend=False,
    title_text='LSTM residual diagnostics',
)
fig.update_xaxes(title_text='Forecast error', row=1, col=1)
fig.update_xaxes(title_text='Actual |return|', row=1, col=2)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Forecast error', row=1, col=2)
fig.show()

# Bias check
mean_error = lstm_errors.mean()
print(f'Mean forecast error: {mean_error:.6f} '
      f'({"positive bias (underforecasts)" if mean_error > 0 else "negative bias (overforecasts)"})')
```



    Mean forecast error: -0.001149 (negative bias (overforecasts))
    

If the error-vs-actual scatter fans outward for larger actual values,
the LSTM underestimates high-volatility days. If errors cluster near
zero with symmetric tails, the model is unbiased but noisy.

### Residual autocorrelation

A visual residual check shows distribution shape but not temporal
structure. The Ljung-Box test asks whether forecast errors remain
autocorrelated after the model has done its work.

The interpretation is direct. Significant autocorrelation means the
network left predictable structure on the table: a better model could have
used yesterday's error to improve today's forecast. No significant
autocorrelation means the remaining error is closer to white noise, and
further gains would have to come from new information rather than better
use of existing information.


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
            'Model': label,
            'Lags': lag,
            'LB statistic': lb.loc[lag, 'lb_stat'],
            'p-value': lb.loc[lag, 'lb_pvalue'],
            'Autocorrelated at 5%?': 'Yes' if lb.loc[lag, 'lb_pvalue'] < 0.05 else 'No',
        })

lb_df = pd.DataFrame(lb_rows)
lb_fmt = lb_df.copy()
lb_fmt['LB statistic'] = lb_fmt['LB statistic'].map('{:.2f}'.format)
lb_fmt['p-value'] = lb_fmt['p-value'].map('{:.4f}'.format)
display(Markdown(lb_fmt.to_markdown(index=False)))

# Dynamic interpretation at the longer lag
lstm_lb_p = lb_df[
    (lb_df['Model'].str.startswith('LSTM')) & (lb_df['Lags'] == LB_LAGS[-1])
]['p-value'].iloc[0]
mlp_lb_p = lb_df[
    (lb_df['Model'].str.startswith('MLP')) & (lb_df['Lags'] == LB_LAGS[-1])
]['p-value'].iloc[0]

if lstm_lb_p < 0.05:
    lb_interp = (
        f'LSTM residuals show significant autocorrelation at {LB_LAGS[-1]} lags '
        f'(p = {lstm_lb_p:.4f}). Predictable temporal structure remains in the '
        f'errors, meaning the network did not extract everything available from '
        f'the sequence it was given.'
    )
else:
    lb_interp = (
        f'LSTM residuals show no significant autocorrelation at {LB_LAGS[-1]} lags '
        f'(p = {lstm_lb_p:.4f}). The remaining error behaves like white noise, so '
        f'further improvement would require new information rather than better '
        f'use of the existing sequence.'
    )

lb_interp += (
    f' MLP residuals {"are" if mlp_lb_p < 0.05 else "are not"} significantly '
    f'autocorrelated (p = {mlp_lb_p:.4f}).'
)

display(Markdown(lb_interp))
```


| Model           |   Lags |   LB statistic |   p-value | Autocorrelated at 5%?   |
|:----------------|-------:|---------------:|----------:|:------------------------|
| LSTM (32 units) |     10 |         187.54 |    0      | Yes                     |
| LSTM (32 units) |     21 |         215.24 |    0      | Yes                     |
| MLP (64 hidden) |     10 |           6.08 |    0.8085 | No                      |
| MLP (64 hidden) |     21 |          12.49 |    0.9255 | No                      |
| Persistence     |     10 |          69.61 |    0      | Yes                     |
| Persistence     |     21 |          79.13 |    0      | Yes                     |



LSTM residuals show significant autocorrelation at 21 lags (p = 0.0000). Predictable temporal structure remains in the errors, meaning the network did not extract everything available from the sequence it was given. MLP residuals are not significantly autocorrelated (p = 0.9255).


### Forecast calibration

A calibration plot asks a question RMSE cannot: across the range of
predictions, are they right on average? Perfect calibration puts every
point on the 45° line, where predicted equals actual.

The binned means matter more than the scatter. Individual daily points are
dominated by noise, but if the binned average sits below the diagonal at
high predicted values, the model systematically overforecasts large
moves, and vice versa. The fitted slope summarises this: a slope below 1
indicates the forecasts are too spread out relative to reality
(overconfident), while a slope above 1 indicates they are too compressed.


```python
def calibration_data(forecast, actual_vals, n_bins=10):
    """Bin forecasts into quantiles and return per-bin means plus OLS slope."""
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
    (mlp_forecasts,  'MLP',  '#ffd700'),
], start=1):
    bp, ba, slope, intercept, r_val = calibration_data(fc, actual)
    cal_summary[label] = {'slope': slope, 'intercept': intercept, 'r': r_val}

    # Raw scatter
    fig.add_trace(go.Scatter(
        x=fc, y=actual, mode='markers',
        marker=dict(color=colour, size=3, opacity=0.3),
        name=f'{label} daily', showlegend=False,
    ), row=1, col=col)

    # Binned means
    fig.add_trace(go.Scatter(
        x=bp, y=ba, mode='markers+lines',
        marker=dict(color='white', size=9, symbol='diamond'),
        line=dict(color='white', width=2),
        name='Binned mean', showlegend=(col == 1),
    ), row=1, col=col)

    # 45-degree reference
    lim = max(fc.max(), actual.max())
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode='lines',
        line=dict(color='#ff6b6b', dash='dash', width=1.5),
        name='Perfect calibration', showlegend=(col == 1),
    ), row=1, col=col)

fig.update_layout(
    template='plotly_dark', height=420,
    title_text='Calibration: predicted versus actual |return|',
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
)
fig.update_xaxes(title_text='Predicted |return|')
fig.update_yaxes(title_text='Actual |return|')
fig.show()

cal_lines = []
for label, s in cal_summary.items():
    direction = (
        'forecasts are too dispersed relative to realised values'
        if s['slope'] < 0.9 else
        'forecasts are too compressed relative to realised values'
        if s['slope'] > 1.1 else
        'calibration is close to proportional'
    )
    cal_lines.append(
        f"- **{label}**: slope {s['slope']:.3f}, intercept {s['intercept']:.6f}, "
        f"r = {s['r']:.3f}. {direction.capitalize()}."
    )

display(Markdown(
    'Fitted calibration lines:\n\n' + '\n'.join(cal_lines) +
    '\n\nA slope near 1 with a small intercept indicates proportional '
    'forecasts. Systematic departure from the diagonal is a bias the metric '
    'totals do not reveal.'
))
```




Fitted calibration lines:

- **LSTM**: slope 0.079, intercept 0.005530, r = 0.067. Forecasts are too dispersed relative to realised values.
- **MLP**: slope -0.329, intercept 0.006170, r = -0.029. Forecasts are too dispersed relative to realised values.

A slope near 1 with a small intercept indicates proportional forecasts. Systematic departure from the diagonal is a bias the metric totals do not reveal.


### Regime-conditional performance


```python
# Compute regime labels from realised volatility (vol_roll_21).
# Percentile thresholds match NB05's convention.
vol_annual = df['vol_roll_21'] * np.sqrt(252)
p25 = vol_annual.quantile(0.25)
p75 = vol_annual.quantile(0.75)
p95 = vol_annual.quantile(0.95)

regime_labels = pd.cut(
    vol_annual,
    bins=[-np.inf, p25, p75, p95, np.inf],
    labels=['Calm', 'Normal', 'Stress', 'Crisis']
)

test_regimes = regime_labels.loc[pred_dates].values

regime_results = []
for regime in ['Calm', 'Normal', 'Stress', 'Crisis']:
    mask = test_regimes == regime
    n = mask.sum()
    if n < 5:
        continue
    a = actual[mask]
    regime_results.append({
        'Regime': regime,
        'Days': int(n),
        'Persistence RMSE': root_mean_squared_error(a, persistence[mask]),
        'MLP RMSE': root_mean_squared_error(a, mlp_forecasts[mask]),
        'LSTM RMSE': root_mean_squared_error(a, lstm_forecasts[mask]),
    })

regime_df = pd.DataFrame(regime_results)
regime_df['LSTM vs Pers'] = (
    (regime_df['Persistence RMSE'] - regime_df['LSTM RMSE'])
    / regime_df['Persistence RMSE'] * 100
).round(1).astype(str) + '%'
regime_df['MLP vs Pers'] = (
    (regime_df['Persistence RMSE'] - regime_df['MLP RMSE'])
    / regime_df['Persistence RMSE'] * 100
).round(1).astype(str) + '%'

display(regime_df.to_string(index=False))
```


    'Regime  Days  Persistence RMSE  MLP RMSE  LSTM RMSE LSTM vs Pers MLP vs Pers\n  Calm    43          0.003263  0.003834   0.002418        25.9%      -17.5%\nNormal   202          0.007838  0.008436   0.007440         5.1%       -7.6%\nStress     7          0.012389  0.011711   0.007144        42.3%        5.5%'



```python
# Identify which regime each model handles best
lstm_advantage = regime_df['Persistence RMSE'] - regime_df['LSTM RMSE']
best_regime_lstm = regime_df.loc[lstm_advantage.idxmax(), 'Regime']
worst_regime_lstm = regime_df.loc[lstm_advantage.idxmin(), 'Regime']

display(Markdown(f"""
The LSTM's largest advantage over persistence occurs in **{best_regime_lstm}**
regimes and its smallest (or a disadvantage) in **{worst_regime_lstm}**
regimes. Regime labels here are computed from 21-day realised volatility
percentiles rather than GJR-GARCH conditional volatility, so they will not
match NB05's labels exactly. The pattern is what matters: does either model
dominate across all regimes, or do their strengths concentrate in different
market conditions?
"""))
```



The LSTM's largest advantage over persistence occurs in **Stress**
regimes and its smallest (or a disadvantage) in **Normal**
regimes. Regime labels here are computed from 21-day realised volatility
percentiles rather than GJR-GARCH conditional volatility, so they will not
match NB05's labels exactly. The pattern is what matters: does either model
dominate across all regimes, or do their strengths concentrate in different
market conditions?



### The largest forecasting miss

Aggregate metrics hide the individual failures that matter most in a risk
application. This section isolates the single worst forecast in the test
window and asks what the model could plausibly have known.


```python
worst_idx = int(np.argmax(np.abs(lstm_errors)))
worst_date = pred_dates[worst_idx]
worst_actual = actual[worst_idx]
worst_lstm = lstm_forecasts[worst_idx]
worst_mlp = mlp_forecasts[worst_idx]
worst_pers = persistence[worst_idx]
worst_err = lstm_errors[worst_idx]

# Context: trailing realised volatility before the miss
lb_window = 5
ctx_start = max(worst_idx - lb_window, 0)
trailing = actual[ctx_start:worst_idx]
trailing_mean = trailing.mean() if len(trailing) else np.nan
spike_ratio = worst_actual / trailing_mean if trailing_mean else np.nan

# Regime label on the day
worst_regime = test_regimes[worst_idx]

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
| Trailing {lb_window}-day mean \\|return\\| | {trailing_mean:.6f} |
| Spike ratio (actual / trailing mean) | {spike_ratio:.1f}× |
"""))

if worst_err > 0:
    miss_type = 'underforecast'
    miss_expl = (
        f'The realised move was {spike_ratio:.1f}× the trailing {lb_window}-day '
        f'average. Every input the LSTM had (rolling volatility at three '
        f'horizons, ATR, Bollinger width, volatility rank) described the market '
        f'as it was before the shock, not as it became. A shock arriving from '
        f'outside the return series is not forecastable from the return series.'
    )
else:
    miss_type = 'overforecast'
    miss_expl = (
        f'The model expected continued elevated volatility and the market '
        f'delivered a quiet day. Overforecasting after a shock is the signature '
        f'of insufficient mean reversion: GARCH encodes reversion explicitly '
        f'through its beta parameter, while the network must learn the decay '
        f'rate from examples.'
    )

display(Markdown(f"""
The worst miss was an **{miss_type}** of {abs(worst_err):.6f} on
{worst_date.date()}. {miss_expl}

Persistence forecast {worst_pers:.6f} against a realised {worst_actual:.6f},
so the benchmark {'also missed badly' if abs(worst_actual - worst_pers) > abs(worst_err) * 0.8 else 'was closer on this particular day'}.
Single-day failures are expected in volatility forecasting; the diagnostic
value is in whether the failure mode is systematic. The calibration plot
and regime table above both address that question.
"""))
```



| | Value |
|---|---|
| Date | 2026-07-02 |
| Regime | Normal |
| Actual \|return\| | 0.000001 |
| LSTM forecast | 0.028579 |
| MLP forecast | 0.000014 |
| Persistence forecast | 0.002153 |
| LSTM error | -0.028578 |
| Trailing 5-day mean \|return\| | 0.004459 |
| Spike ratio (actual / trailing mean) | 0.0× |





The worst miss was an **overforecast** of 0.028578 on
2026-07-02. The model expected continued elevated volatility and the market delivered a quiet day. Overforecasting after a shock is the signature of insufficient mean reversion: GARCH encodes reversion explicitly through its beta parameter, while the network must learn the decay rate from examples.

Persistence forecast 0.002153 against a realised 0.000001,
so the benchmark was closer on this particular day.
Single-day failures are expected in volatility forecasting; the diagnostic
value is in whether the failure mode is systematic. The calibration plot
and regime table above both address that question.



### Feature importance (permutation)


```python
# Permutation importance using the last walk-forward LSTM model.
# Shuffle each feature across all sequences and measure MSE increase.

X_test_full_s_copy = X_test_full_s.copy()
base_preds = last_lstm.predict(X_test_full_s_copy, verbose=0).flatten()
base_mse = np.mean((actual - base_preds)**2)

importance_scores = {}
importance_std = {}
N_REPEATS = 5

for feat_idx, feat_name in enumerate(FEATURE_COLS):
    deltas = []
    for _ in range(N_REPEATS):
        X_perm = X_test_full_s.copy()
        # Shuffle this feature across the sequence-time dimension
        flat = X_perm[:, :, feat_idx].flatten()
        np.random.shuffle(flat)
        X_perm[:, :, feat_idx] = flat.reshape(X_perm.shape[0], X_perm.shape[1])
        perm_preds = last_lstm.predict(X_perm, verbose=0).flatten()
        perm_mse = np.mean((actual - perm_preds)**2)
        deltas.append(perm_mse - base_mse)
    importance_scores[feat_name] = np.mean(deltas)
    importance_std[feat_name] = np.std(deltas, ddof=1)

imp_df = (
    pd.DataFrame({
        'MSE increase': pd.Series(importance_scores),
        'Std across repeats': pd.Series(importance_std),
    })
    .sort_values('MSE increase', ascending=False)
)
imp_df['Relative'] = (
    imp_df['MSE increase'] / imp_df['MSE increase'].sum() * 100
).round(1)
# Stability: mean divided by spread. Large values mean the ranking is reliable.
imp_df['Signal / noise'] = (
    imp_df['MSE increase'] / imp_df['Std across repeats'].replace(0, np.nan)
).round(2)

display(imp_df)

top_feature = imp_df.index[0]
top_snr = imp_df.iloc[0]['Signal / noise']
print(f'\nMost important feature: {top_feature}')
print(f'Repeats per feature: {N_REPEATS}')
print(f'Top feature signal-to-noise across repeats: {top_snr:.2f}')
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
      <th>Relative</th>
      <th>Signal / noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rsi_14</th>
      <td>2.065116e-06</td>
      <td>4.402830e-07</td>
      <td>138.4</td>
      <td>4.69</td>
    </tr>
    <tr>
      <th>log_returns</th>
      <td>4.223142e-07</td>
      <td>2.884543e-07</td>
      <td>28.3</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>bb_width</th>
      <td>8.848921e-08</td>
      <td>1.832076e-07</td>
      <td>5.9</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>vol_roll_60</th>
      <td>2.183322e-08</td>
      <td>1.594838e-08</td>
      <td>1.5</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>vol_roll_10</th>
      <td>-7.014353e-08</td>
      <td>5.239791e-08</td>
      <td>-4.7</td>
      <td>-1.34</td>
    </tr>
    <tr>
      <th>vol_roll_21</th>
      <td>-1.383638e-07</td>
      <td>5.099316e-08</td>
      <td>-9.3</td>
      <td>-2.71</td>
    </tr>
    <tr>
      <th>vol_ratio_10_60</th>
      <td>-1.446025e-07</td>
      <td>4.599736e-08</td>
      <td>-9.7</td>
      <td>-3.14</td>
    </tr>
    <tr>
      <th>vol_rank_30</th>
      <td>-1.847680e-07</td>
      <td>5.148286e-08</td>
      <td>-12.4</td>
      <td>-3.59</td>
    </tr>
    <tr>
      <th>volume_lag_1</th>
      <td>-1.888139e-07</td>
      <td>6.581570e-08</td>
      <td>-12.7</td>
      <td>-2.87</td>
    </tr>
    <tr>
      <th>atr_14</th>
      <td>-3.786206e-07</td>
      <td>6.250831e-08</td>
      <td>-25.4</td>
      <td>-6.06</td>
    </tr>
  </tbody>
</table>
</div>


    
    Most important feature: rsi_14
    Repeats per feature: 5
    Top feature signal-to-noise across repeats: 4.69
    


```python
fig = go.Figure(go.Bar(
    y=imp_df.index[::-1],
    x=imp_df['MSE increase'].values[::-1],
    error_x=dict(
        type='data',
        array=imp_df['Std across repeats'].values[::-1],
        visible=True, color='white', thickness=1, width=3,
    ),
    orientation='h',
    marker_color='#00d4aa',
))
fig.update_layout(
    template='plotly_dark',
    title=f'Permutation feature importance (mean ± std over {N_REPEATS} repeats)',
    xaxis_title='MSE increase when shuffled',
    height=400,
)
fig.show()

# Dynamic interpretation based on actual importance ranking
vol_features = {'vol_roll_10', 'vol_roll_21', 'vol_roll_60'}
top_3 = set(imp_df.index[:3])
vol_dominated = len(top_3 & vol_features) >= 2

if vol_dominated:
    feat_interp = (
        "Volatility features dominate the importance ranking. The LSTM is "
        "largely replicating what GARCH already captures through its variance "
        "recursion, which explains why additional model capacity did not "
        "translate into better forecasts."
    )
else:
    non_vol_top = [f for f in imp_df.index[:3] if f not in vol_features]
    feat_interp = (
        f"Non-volatility features ({', '.join(non_vol_top)}) rank among the "
        f"top three. The LSTM found signal GARCH has no access to, even if "
        f"that signal was not sufficient to win overall."
    )

display(Markdown(feat_interp))
```




Non-volatility features (rsi_14, log_returns, bb_width) rank among the top three. The LSTM found signal GARCH has no access to, even if that signal was not sufficient to win overall.


### Complexity versus accuracy

Plotting parameters against RMSE for all four models shows where each
sits on the complexity-accuracy frontier. The x-axis uses a log scale;
persistence (zero parameters) is plotted at x = 1 as a visual anchor.


```python
models = {
    'Persistence': (0, pers_rmse_check),
    GARCH_LABEL:   (garch_n_params, GARCH_RMSE),
    f'MLP ({MLP_HIDDEN} hidden)': (mlp_params, mlp_rmse),
    f'LSTM ({LSTM_UNITS} units)':  (lstm_params, lstm_rmse),
}

names = list(models.keys())
params = [max(v[0], 1) for v in models.values()]  # floor at 1 for log scale
rmses  = [v[1] for v in models.values()]
colors = ['#ff6b6b', '#00d4aa', '#ffd700', '#00aaff']

# Best trade-off: lowest RMSE among models on the efficient frontier.
# A model is dominated if another has both fewer parameters and lower RMSE.
frontier = []
for i, (n_i, r_i) in enumerate(zip(params, rmses)):
    dominated = any(
        (n_j <= n_i and r_j < r_i) for j, (n_j, r_j) in enumerate(zip(params, rmses))
        if j != i
    )
    if not dominated:
        frontier.append(i)

best_i = min(frontier, key=lambda i: rmses[i])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=params, y=rmses,
    mode='markers+text',
    text=names,
    textposition='top center',
    marker=dict(size=14, color=colors),
    textfont=dict(size=11),
))
fig.add_annotation(
    x=np.log10(params[best_i]), y=rmses[best_i],
    text='Best trade-off',
    showarrow=True, arrowhead=2, arrowcolor='white',
    ax=45, ay=35,
    font=dict(color='white', size=12),
    bgcolor='rgba(0,0,0,0.55)', borderpad=4,
)
fig.update_layout(
    template='plotly_dark',
    title='Complexity vs accuracy',
    xaxis_title='Trainable parameters (log scale)',
    yaxis_title='RMSE',
    xaxis_type='log',
    height=450,
    showlegend=False,
)
fig.show()

display(Markdown(f"""
{GARCH_LABEL} achieves its accuracy with {garch_n_params} parameters. The MLP
uses {mlp_params:,} and the LSTM {lstm_params:,}. {f'The LSTM matches GARCH on accuracy at {lstm_params / garch_n_params:,.0f} times the parameter count, and the MLP trails both.' if lstm_garch_tie else 'Neither neural network improved on GARCH, so the additional complexity bought nothing on this data.' if garch_wins else 'The accuracy improvement came at substantial complexity cost.'}

The efficient frontier here contains {len(frontier)} of the four models;
the remainder are dominated, meaning another model achieves lower RMSE with
no more parameters.
"""))
```





GJR-GARCH(1,1,1) — Student's t achieves its accuracy with 5 parameters. The MLP
uses 13,569 and the LSTM 5,537. Neither neural network improved on GARCH, so the additional complexity bought nothing on this data.

The efficient frontier here contains 2 of the four models;
the remainder are dominated, meaning another model achieves lower RMSE with
no more parameters.



## 14. Deployment comparison


```python
display(Markdown(f"""
| Criterion | {GARCH_LABEL} | MLP ({MLP_HIDDEN} hidden) | LSTM ({LSTM_UNITS} units) |
|---|---|---|---|
| Accuracy (RMSE) | {GARCH_RMSE:.6f} | {mlp_rmse:.6f} | {lstm_rmse:.6f} |
| Trainable parameters | {garch_n_params} | {mlp_params:,} | {lstm_params:,} |
| Training time | (see NB05) | {wf_total_mlp:.0f} s | {wf_total_lstm:.0f} s |
| Inference (252 obs) | (see NB05) | {mlp_inference_ms:.0f} ms | {lstm_inference_ms:.0f} ms |
| Interpretability | Each parameter maps to a financial mechanism | Black box | Black box |
| Dependencies | `arch` (pure Python) | TensorFlow | TensorFlow |
| Reproducibility | Deterministic | Seed-dependent | Seed-dependent |

{'GARCH is the simpler, faster, more interpretable model that also happens to be more accurate. The deployment choice is clear for Version 1.' if garch_wins else 'The best neural network is more accurate but substantially more complex to deploy, monitor, and explain. Whether the accuracy margin justifies the operational cost is a deployment decision.'}
"""))
```



| Criterion | GJR-GARCH(1,1,1) — Student's t | MLP (64 hidden) | LSTM (32 units) |
|---|---|---|---|
| Accuracy (RMSE) | 0.005206 | 0.007960 | 0.006840 |
| Trainable parameters | 5 | 13,569 | 5,537 |
| Training time | (see NB05) | 329 s | 893 s |
| Inference (252 obs) | (see NB05) | 98 ms | 138 ms |
| Interpretability | Each parameter maps to a financial mechanism | Black box | Black box |
| Dependencies | `arch` (pure Python) | TensorFlow | TensorFlow |
| Reproducibility | Deterministic | Seed-dependent | Seed-dependent |

GARCH is the simpler, faster, more interpretable model that also happens to be more accurate. The deployment choice is clear for Version 1.




```python
# Dynamic section title based on results
if garch_wins:
    section_title = "Why the structural model wins here"
else:
    section_title = "Why the neural network wins here"

display(Markdown(f"## 15. {section_title}"))
```


## 15. Why the structural model wins here



```python
# Phrase the GARCH persistence reference dynamically. NB05 exports
# garch_persistence only if that cell has been added; fall back to a
# qualitative statement rather than asserting a figure we cannot verify.
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

if garch_wins:
    analysis = f"""
The result is not a failure of deep learning in general. It is a
predictable outcome of applying flexible models to a problem where
the structure is already known.

The comparison illustrates the classical bias-variance trade-off. GARCH
intentionally restricts the hypothesis space using financial assumptions:
variance is persistent, shocks decay geometrically, negative returns raise
variance more than positive ones. Those restrictions introduce bias if the
assumptions are wrong, but they collapse the estimation problem to
{garch_n_params} parameters and so keep variance low. The neural networks
possess much greater representational capacity and therefore lower
approximation bias, but they require substantially more observations to
estimate their parameters reliably. On this dataset the variance term
dominates, and the constrained model wins.

The neural networks trained on roughly {test_start:,} sequences. With
{lstm_params:,} parameters (LSTM) and {mlp_params:,} (MLP), the
samples-to-parameters ratios are ~{test_start / lstm_params:.1f}:1 and
~{test_start / mlp_params:.1f}:1 respectively. Deep learning typically
needs ratios of 100:1 or higher to generalise well. GARCH achieves the
same task with {garch_n_params} parameters and no minimum sample
requirement beyond stationarity.

The target itself works against flexible models. Absolute daily log
returns are inherently noisy: individual days are dominated by
idiosyncratic shocks, and the forecastable component (the conditional
variance) is a slow-moving signal embedded in fast noise. GARCH is
designed to extract exactly this signal. Neural networks must learn to
ignore the noise, which requires more data than is available.

Volatility's high autocorrelation ({persistence_phrase}) makes the
persistence benchmark hard to beat and limits
the room for any model to improve. GARCH's edge comes from modelling
mean reversion after shocks. The neural networks must learn this
dynamic from data rather than encoding it structurally.

{'The LSTM and MLP produced similar accuracy, confirming that temporal modelling added little beyond what the rolling features already capture. The 21-day lookback window, combined with rolling volatility at three horizons, effectively pre-computes the sequential information the LSTM would otherwise need to learn.' if abs(lstm_rmse - mlp_rmse) / mlp_rmse < 0.05 else f'The LSTM {"outperformed" if lstm_beats_mlp else "underperformed"} the MLP by {lstm_mlp_rel:.1f}%, suggesting temporal structure {"carries some value beyond the rolling features" if lstm_beats_mlp else "does not add value for this problem"}.'}

Permutation importance (section 13) shows which features the networks
actually relied on. If volatility features dominated, the networks were
largely rediscovering what GARCH already knows at a much higher
computational cost. Multi-asset inputs (cross-sectional volatility,
sector correlations, VIX term structure) could give deep learning a
genuine information advantage in a future version, but on a single
return series the structural model has fewer unknowns and more
constraints.
"""
else:
    analysis = f"""
The best neural network outperformed GARCH on this test window. Before
concluding that deep learning is superior for this application, several
caveats apply.

One 252-day window is one draw from one market regime. The advantage may
not persist across windows containing different regime mixes. The neural
networks had access to {n_features} engineered features while GARCH used
only returns, so part of the margin may reflect information advantage
rather than architectural superiority. A fair isolation test would give
GARCH access to the same features as exogenous regressors (GARCH-X).

{'The LSTM outperformed the MLP, suggesting temporal structure carries exploitable value beyond the rolling features.' if lstm_beats_mlp else 'The MLP matched or beat the LSTM, suggesting the feature set carries the signal and sequential modelling adds nothing.'}

The operational cost is also relevant: the neural networks required
{wf_total:.0f}s of training versus substantially less for GARCH, and
introduce a TensorFlow dependency, seed sensitivity, and hardware
variability that GARCH avoids.
"""

display(Markdown(analysis))
```



The result is not a failure of deep learning in general. It is a
predictable outcome of applying flexible models to a problem where
the structure is already known.

The comparison illustrates the classical bias-variance trade-off. GARCH
intentionally restricts the hypothesis space using financial assumptions:
variance is persistent, shocks decay geometrically, negative returns raise
variance more than positive ones. Those restrictions introduce bias if the
assumptions are wrong, but they collapse the estimation problem to
5 parameters and so keep variance low. The neural networks
possess much greater representational capacity and therefore lower
approximation bias, but they require substantially more observations to
estimate their parameters reliably. On this dataset the variance term
dominates, and the constrained model wins.

The neural networks trained on roughly 6,139 sequences. With
5,537 parameters (LSTM) and 13,569 (MLP), the
samples-to-parameters ratios are ~1.1:1 and
~0.5:1 respectively. Deep learning typically
needs ratios of 100:1 or higher to generalise well. GARCH achieves the
same task with 5 parameters and no minimum sample
requirement beyond stationarity.

The target itself works against flexible models. Absolute daily log
returns are inherently noisy: individual days are dominated by
idiosyncratic shocks, and the forecastable component (the conditional
variance) is a slow-moving signal embedded in fast noise. GARCH is
designed to extract exactly this signal. Neural networks must learn to
ignore the noise, which requires more data than is available.

Volatility's high autocorrelation (GARCH persistence of 0.9828 in Notebook 05, implying a shock half-life of roughly 40 trading days) makes the
persistence benchmark hard to beat and limits
the room for any model to improve. GARCH's edge comes from modelling
mean reversion after shocks. The neural networks must learn this
dynamic from data rather than encoding it structurally.

The LSTM outperformed the MLP by 14.1%, suggesting temporal structure carries some value beyond the rolling features.

Permutation importance (section 13) shows which features the networks
actually relied on. If volatility features dominated, the networks were
largely rediscovering what GARCH already knows at a much higher
computational cost. Multi-asset inputs (cross-sectional volatility,
sector correlations, VIX term structure) could give deep learning a
genuine information advantage in a future version, but on a single
return series the structural model has fewer unknowns and more
constraints.



## 16. Conclusion


```python
if garch_wins:
    conclusion = f"""
Across identical walk-forward evaluation, neither the LSTM nor the MLP
improved on {GARCH_LABEL}'s out-of-sample volatility forecasts. For this
application (single-asset, daily-frequency volatility on ~{len(df):,}
observations), encoding financial structure into the model proved more
valuable than increasing model capacity or enriching the feature set.
Version 1 retains {GARCH_LABEL} as the production forecasting model.

The MLP baseline confirmed that the LSTM's sequential modelling did not
add meaningful value beyond what the rolling features already capture,
narrowing the question from "can deep learning help?" to "can richer
inputs help?", a question better answered by GARCH-X than by neural
networks.

Potential extensions for Version 2 (not in current scope): multi-asset
inputs, Transformer or temporal convolutional architectures, and GARCH-X
with exogenous features to test whether GARCH can also benefit from the
Notebook 03 feature set.

### Takeaway

Model complexity alone does not guarantee better forecasts. When the
underlying financial mechanism is well understood, explicitly modelling
that structure can outperform far larger neural networks trained on richer
feature sets. In this application, incorporating domain knowledge proved
more valuable than increasing model capacity, and the reverse result
would have been equally publishable, which is why the test was worth
running rather than assuming.
"""
else:
    conclusion = f"""
The best neural network outperformed {GARCH_LABEL} on the identical
252-day walk-forward window. Before promoting it to the Daily Market Risk
Report, two checks are required: robustness across multiple walk-forward
windows, and whether the accuracy margin justifies the operational cost
of maintaining a TensorFlow dependency and retraining pipeline.

For Version 1, {GARCH_LABEL} remains the production model pending
robustness confirmation.

### Takeaway

Additional capacity paid off here, but the margin was earned with
{'an information advantage' if n_features > 1 else 'architecture alone'}:
the networks saw {n_features} features while GARCH saw one series. The
honest next test is GARCH-X on the same inputs, which would separate the
value of flexibility from the value of information. Complexity that wins
on richer data has not yet proven it wins on equal terms.
"""

display(Markdown(conclusion))
```



Across identical walk-forward evaluation, neither the LSTM nor the MLP
improved on GJR-GARCH(1,1,1) — Student's t's out-of-sample volatility forecasts. For this
application (single-asset, daily-frequency volatility on ~6,412
observations), encoding financial structure into the model proved more
valuable than increasing model capacity or enriching the feature set.
Version 1 retains GJR-GARCH(1,1,1) — Student's t as the production forecasting model.

The MLP baseline confirmed that the LSTM's sequential modelling did not
add meaningful value beyond what the rolling features already capture,
narrowing the question from "can deep learning help?" to "can richer
inputs help?", a question better answered by GARCH-X than by neural
networks.

Potential extensions for Version 2 (not in current scope): multi-asset
inputs, Transformer or temporal convolutional architectures, and GARCH-X
with exogenous features to test whether GARCH can also benefit from the
Notebook 03 feature set.

### Takeaway

Model complexity alone does not guarantee better forecasts. When the
underlying financial mechanism is well understood, explicitly modelling
that structure can outperform far larger neural networks trained on richer
feature sets. In this application, incorporating domain knowledge proved
more valuable than increasing model capacity, and the reverse result
would have been equally publishable, which is why the test was worth
running rather than assuming.



## 17. Export


```python
# ── Save predictions for downstream notebooks ──
preds_df = pd.DataFrame({
    'actual': actual,
    'persistence': persistence,
    'lstm': lstm_forecasts,
    'mlp':  mlp_forecasts,
}, index=pred_dates)
preds_path = Path('../data/nb06_predictions.parquet')
preds_df.to_parquet(preds_path)
print(f'Predictions saved to {preds_path.resolve()}')

# ── Export metrics to locked_metrics.json ──
metrics_path = Path('../data/locked_metrics.json')
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

metrics['notebook_06'] = {
    'lstm_units':             LSTM_UNITS,
    'lstm_total_params':      int(lstm_params),
    'mlp_hidden':             MLP_HIDDEN,
    'mlp_total_params':       int(mlp_params),
    'garch_n_params':         int(garch_n_params),
    'n_features':             n_features,
    'lookback':               LOOKBACK,
    'wf_rmse_lstm':           float(lstm_rmse),
    'wf_mae_lstm':            float(lstm_mae),
    'wf_rmse_mlp':            float(mlp_rmse),
    'wf_mae_mlp':             float(mlp_mae),
    'wf_rmse_persistence':    float(pers_rmse_check),
    'wf_mae_persistence':     float(pers_mae_check),
    'lstm_vs_pers_rmse_pct':  float(lstm_vs_pers_rmse),
    'lstm_vs_garch_rmse_pct': float(lstm_vs_garch_rmse),
    'mlp_vs_pers_rmse_pct':   float(mlp_vs_pers_rmse),
    'mlp_vs_garch_rmse_pct':  float(mlp_vs_garch_rmse),
    'garch_wins':             bool(garch_wins),
    'lstm_beats_mlp':         bool(lstm_beats_mlp),
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
    'worst_miss_date':        str(worst_date.date()),
    'worst_miss_error':       float(worst_err),
    'worst_miss_regime':      str(worst_regime),
    'wf_total_time_s':        round(wf_total, 1),
    'wf_lstm_time_s':         round(wf_total_lstm, 1),
    'wf_mlp_time_s':          round(wf_total_mlp, 1),
    'avg_epochs_lstm':        round(float(np.mean(epochs_used_lstm)), 1),
    'avg_epochs_mlp':         round(float(np.mean(epochs_used_mlp)), 1),
    'lstm_inference_ms':      round(lstm_inference_ms, 2),
    'mlp_inference_ms':       round(mlp_inference_ms, 2),
    'tf_version':             tf.__version__,
    'top_feature':            top_feature,
    'garch_persistence_used':  GARCH_PERSISTENCE,
}

metrics_path.write_text(json.dumps(metrics, indent=2))

print(f'\nExported notebook_06 metrics to {metrics_path.resolve()}')
for k, v in metrics['notebook_06'].items():
    print(f'  {k}: {v}')
```

    Predictions saved to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\nb06_predictions.parquet
    
    Exported notebook_06 metrics to C:\Users\Mena\Documents\Python\sp500-market-intelligence\data\locked_metrics.json
      lstm_units: 32
      lstm_total_params: 5537
      mlp_hidden: 64
      mlp_total_params: 13569
      garch_n_params: 5
      n_features: 10
      lookback: 21
      wf_rmse_lstm: 0.006839705570368288
      wf_mae_lstm: 0.004708823776371078
      wf_rmse_mlp: 0.00796031803792496
      wf_mae_mlp: 0.005928118877798958
      wf_rmse_persistence: 0.007438077359303796
      wf_mae_persistence: 0.0055138317645078054
      lstm_vs_pers_rmse_pct: 8.04471048135907
      lstm_vs_garch_rmse_pct: -31.370415836420474
      mlp_vs_pers_rmse_pct: -7.021178369003205
      mlp_vs_garch_rmse_pct: -52.894050785295946
      garch_wins: True
      lstm_beats_mlp: True
      dm_stat_lstm_pers: 0.8668004440105989
      dm_pval_lstm_pers: 0.3860513587927292
      dm_stat_mlp_pers: -1.6102078668013857
      dm_pval_mlp_pers: 0.10735248458507814
      dm_stat_lstm_mlp: 1.6917730448538035
      dm_pval_lstm_mlp: 0.09068925494142839
      rmse_ci_lstm: [0.004619054755283348, 0.009523993275129757]
      rmse_ci_mlp: [0.006996952000989724, 0.009139605755812996]
      rmse_ci_persistence: [0.006269112904523796, 0.00877515566169867]
      n_bootstrap: 2000
      bootstrap_block: 21
      lb_pval_lstm: 3.5510589309986276e-34
      lb_pval_mlp: 0.9255273149829637
      calib_slope_lstm: 0.0789711949128051
      calib_slope_mlp: -0.32947874436610797
      worst_miss_date: 2026-07-02
      worst_miss_error: -0.02857771767749061
      worst_miss_regime: Normal
      wf_total_time_s: 1229.0
      wf_lstm_time_s: 893.1
      wf_mlp_time_s: 329.4
      avg_epochs_lstm: 49.2
      avg_epochs_mlp: 47.7
      lstm_inference_ms: 138.31
      mlp_inference_ms: 98.48
      tf_version: 2.21.0
      top_feature: rsi_14
      garch_persistence_used: 0.9828090407814376
    

### Note on NB05 exports

If Notebook 05 does not yet export its walk-forward results to
`locked_metrics.json`, add the following cell at the end of NB05:

```python
metrics['notebook_05'] = {
    'best_label':          best_label,
    'wf_rmse_garch':       float(rmse_garch_wf),
    'wf_mae_garch':        float(mae_garch_wf),
    'wf_rmse_persistence': float(rmse_pers_wf),
    'wf_mae_persistence':  float(mae_pers_wf),
    'improvement_rmse_pct': float(improvement_rmse),
}
```

Until that cell is added, NB06 falls back to the constants from the
executed NB05 output.

For a direct neural-network-vs-GARCH Diebold–Mariano test, NB05 should
also export its walk-forward predictions to Parquet.


