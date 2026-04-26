# Development Log — LSTM Rainfall-Runoff Proof of Concept

Troubleshooting and refactoring record for `scripts/lstm_rainfall_runoff02.ipynb`.

---

## Issue 1 — Three Nested Git Repositories

| | |
|---|---|
| **Problem** | Antigravity (git GUI) showed three separate repos: `LSTM`, `LSTM-Mini-Project`, `scripts` |
| **Root Cause** | `git init` was run independently in three folders (`LSTM/`, `LSTM/scripts/`, and GitHub clone created `LSTM/LSTM-Mini-Project/`), all pointing to the same remote |
| **Fix** | Deleted `.git` from `LSTM/` and `scripts/`, moved all code files into `LSTM-Mini-Project/scripts/`, force-pushed to clean up the stray "try" commit from GitHub |
| **Result** | Single clean repo with one `.git` — only `LSTM-Mini-Project` shows in git GUI |

---

## Issue 2 — Validation Set Sliding Window Leakage

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

## Issue 3 — `create_sequences` Defined as Nested Function

| | |
|---|---|
| **Problem** | `create_sequences` was defined inside `prepare_data`, making it impossible to call independently |
| **Root Cause** | Original code bundled the val split inside `prepare_data` so there was no need to call it separately — once Issue 2 was fixed, the function needed to be called twice from outside |
| **Fix** | Moved `create_sequences` out as a standalone top-level function |
| **Result** | Reusable and readable; called separately for train, val, and test splits |

---

## Issue 4 — `train_model` Returning the Optimizer

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

## Issue 5 — Overfitting

| | |
|---|---|
| **Problem** | Train loss kept decreasing while val loss plateaued, with a 10–17× gap between them |
| **Root Cause** | Single-basin LSTM with sufficient capacity to memorize 20 years of training-period hydrology; no regularization or stopping mechanism |
| **Expected Fix** | Three countermeasures combined: Weight Decay, ReduceLROnPlateau, Early Stopping |
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
# EPOCHS=100 is a ceiling; Early Stopping terminates before it
```

---

## Issue 6 — Jupyter Overwrote File Edits

| | |
|---|---|
| **Problem** | Code edits made to the `.ipynb` file were invisible after reopening Jupyter |
| **Root Cause** | Jupyter autosaves the entire notebook (including all cell sources) when cells are executed. If the notebook was already open with old cell content in memory, running it caused Jupyter to overwrite the updated file on disk with the old version |
| **Fix** | Restored the correct version from the git commit (`git restore scripts/lstm_rainfall_runoff02.ipynb`), then reopened the file fresh in Jupyter (File → Revert Notebook) before running |
| **Lesson** | After editing a `.ipynb` file externally, always close and reopen it in Jupyter (or File → Revert) before running — never trust the in-memory state |

---

## Final Metrics (Basin 13340600 — NF Clearwater River, ID)

| Metric | Before refactor | After all fixes | Threshold |
|---|---|---|---|
| NSE | 0.744 | **0.773** | > 0.70 ✓ |
| KGE | 0.849 | **0.870** | > 0.70 ✓ |
| RMSE | 1.333 mm/day | **1.257 mm/day** | lower ✓ |
| PBIAS | 6.83% | **4.55%** | < ±10% ✓ |
| val_loss (best) | ~0.41 | **~0.18** | — |
