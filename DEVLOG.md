# Development Log — LSTM-GR4J Rainfall-Runoff Proof of Concept

This repository contains a proof-of-concept LSTM model for rainfall-runoff modeling using the CAMELS-US dataset. The model is trained on a single basin (13340600) and evaluated using NSE, KGE, RMSE, and PBIAS metrics. The repository also includes a GR4J baseline model and a simple linear regression baseline for comparison. All models are trained and evaluated using the same train/validation/test split.

Troubleshooting and refactoring record for `scripts/lstm_gr4j_rainfall_runoff.ipynb`.

---

## Issue 1 — Validation Set Sliding Window Leakage

| | |
|---|---|
| **Problem** | `val_loss` was artificially high (~0.41), making overfitting look worse than it was |
| **Root Cause** | Sequences were created from the full training period first, then split into train/val after. Because each sequence is 365 days long, sequences near the boundary shared up to 364 days of overlap — the val set was not truly independent |
| **Fix** | Split at the **date level** first (`train_raw` / `val_raw`), then called `create_sequences()` separately on each split. Scalers are also now fit only on `train_raw` |
| **Result** | `val_loss` dropped from ~0.41 to ~0.18 — a much more honest picture of validation performance |

```python
# Before (leaky)
X_all, y_all = create_sequences(all_scaled, seq_len)
X_val = X_all[-val_size:]   # shares 364 days with X_train at boundary

# After (clean)
train_raw = train_data.iloc[:val_cutoff]
val_raw   = train_data.iloc[val_cutoff:]
X_train, y_train = create_sequences(feature_scaler.fit_transform(train_raw), ...)
X_val,   y_val   = create_sequences(feature_scaler.transform(val_raw), ...)
```

---

## Issue 2 — `create_sequences` Defined as Nested Function

| | |
|---|---|
| **Problem** | `create_sequences` was defined inside `prepare_data`, making it impossible to call independently |
| **Root Cause** | Original code bundled the val split inside `prepare_data` so there was no need to call it separately — once Issue 1 was fixed, the function needed to be called from outside |
| **Fix** | Moved `create_sequences` out as a standalone top-level function |
| **Result** | Reusable and readable; called separately for train, val, and test splits |

---

## Issue 3 — `train_model` Returning the Optimizer

| | |
|---|---|
| **Problem** | `train_model` created the optimizer internally and returned it alongside losses, making the function signature unintuitive |
| **Root Cause** | Optimizer state was needed for checkpoint saving, so it was returned as a workaround |
| **Fix** | `optimizer` and `scheduler` are now created outside and passed in as parameters. Checkpoint saving accesses `optimizer.state_dict()` directly |
| **Result** | `train_model` returns only `(train_losses, val_losses)` — clean and reusable |

```python
# Before
def train_model(model, ..., lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ...
    return train_losses, val_losses, optimizer   # unintuitive

# After
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, ...)

def train_model(model, ..., optimizer, scheduler, device, patience=15):
    ...
    return train_losses, val_losses              # clean
```

---

## Issue 4 — Overfitting

| | |
|---|---|
| **Problem** | Train loss kept decreasing while val loss plateaued, with a 10–17× gap between them |
| **Root Cause** | Single-basin LSTM with sufficient capacity to memorize 20 years of training-period hydrology; no regularization or stopping mechanism |
| **Fix** | Three countermeasures applied together: Weight Decay, ReduceLROnPlateau, Early Stopping |
| **Result** | Train/val gap narrowed; NSE improved from 0.744 → 0.773 |

**Countermeasures applied:**

| Technique | Setting | Purpose |
|---|---|---|
| Weight Decay | `1e-4` in Adam | L2 penalty — discourages memorizing training noise |
| ReduceLROnPlateau | `patience=8, factor=0.5` | Halves LR when val_loss stagnates — finer convergence |
| Early Stopping | `patience=15` | Stops training if val_loss shows no improvement for 15 epochs |

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=8, factor=0.5
)
# EPOCHS=50 is a ceiling; Early Stopping terminates before it
```

---

## Issue 5 — GR4J Parameter Saturation at Narrow Bounds

| | |
|---|---|
| **Problem** | The GR4J + snow calibrator drove X1 (production capacity) and X3 (routing capacity) hard against the upper edge of the search bounds, producing a non-physical "ceiling" parameter set and a worse fit on this snow-dominated PNW basin |
| **Root Cause** | The initial GR4J core bounds were too narrow for deep-storage snow-dominated basins like 13340600 — the optimum lives outside that box, so the calibrator could only press against the wall |
| **Fix** | Widened GR4J core bounds to Perrin et al. (2003) ranges + safety margin: X1 (10, 2000), X2 (-10, 10), X3 (10, 500), X4 (0.5, 10). Snow parameter bounds (t_snow, t_melt, ddf) deliberately kept moderate so the snow module stays identifiable rather than absorbing GR4J routing dynamics |
| **Result** | Calibrator found a deeper, physically realistic production store: **X1 = 889.66 mm** (no longer saturated), **X3 = 402.39 mm**. Train NSE = 0.7748, Test NSE = 0.6394 |

```python
# gr4j.py — DEFAULT_BOUNDS_SNOW (current, widened)
DEFAULT_BOUNDS_SNOW = [
    ( 10.0, 2000.0),  # X1     production capacity (mm)
    (-10.0,   10.0),  # X2     exchange coefficient (mm/day)
    ( 10.0,  500.0),  # X3     routing capacity (mm)
    (  0.5,   10.0),  # X4     UH time base (days)
    ( -3.0,    3.0),  # t_snow phase threshold (degC)
    (  0.0,    5.0),  # t_melt melt onset (degC)
    (  1.0,    8.0),  # ddf    degree-day factor (mm/degC/day)
]
```

**Calibrated parameters (basin 13340600):**

| Parameter | Value | Meaning |
|---|---|---|
| X1 | 889.66 mm | Production store capacity |
| X2 | 5.371 mm/day | Groundwater exchange coefficient |
| X3 | 402.39 mm | Routing store capacity |
| X4 | 1.250 days | UH time base |
| t_snow | -1.856 °C | Phase threshold (snow ↔ rain) |
| t_melt | 2.195 °C | Melt onset threshold |
| ddf | 2.467 mm/°C/day | Degree-day melt factor |

---

## Final Metrics (Basin 13340600 — NF Clearwater River, ID)

LSTM shows the before/after of all fixes (Issues 1–4). GR4J + snow uses the widened bounds from Issue 5. Linear regression is the floor baseline. All three are evaluated on the same test period (2000-10-01 → 2010-09-30) and the same LSTM-aligned dates (3287 days).

| Metric | LSTM (before) | LSTM (after fixes) ✓ | GR4J + Snow | Linear (floor) | Threshold |
|---|---|---|---|---|---|
| NSE | 0.744 | **0.773** | 0.639 | 0.390 | > 0.70 |
| KGE | 0.849 | **0.870** | 0.731 | 0.434 | > 0.70 |
| RMSE (mm/day) | 1.333 | **1.257** | 1.582 | 2.058 | lower |
| PBIAS | 6.83% | **4.55%** | 0.86% | 2.83% | < ±10% |
| val_loss (best) | ~0.41 | **~0.18** | — | — | — |

**Headline:** LSTM beats GR4J by **+0.133 NSE** and the linear floor by **+0.383 NSE**, justifying the data-driven complexity. GR4J still beats the linear floor by **+0.249 NSE**, confirming the physical-based baseline is meaningful (not trivial). GR4J's lower PBIAS (0.86%) reflects its mass-conservation structure, while the LSTM trades a small bias for stronger correlation and variability matching.
