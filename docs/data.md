# Preparing data for rnns-jax

This page describes the data format `fit` expects and how to get there from a table or an array. It's written to be followed step by step, by a person or by an AI coding assistant preparing data in another project.

For a runnable version, see [examples/quickstart.ipynb](../examples/quickstart.ipynb).

## What the models expect

| Name | Shape | Meaning |
|---|---|---|
| `X` | `(N, window, F_in)` | N samples. Each is `window` consecutive time steps with `F_in` features. |
| `y` | `(N, F_out)` | What to predict for each sample: the row `horizon` steps after the window ends. |

- Build the model with `in_ftrs=F_in` and `out_ftrs=F_out`. If they don't match the data, `fit` stops on the first batch with an error that shows both shapes.
- The loss is MSE, so targets must be continuous numbers (regression or forecasting).
- `X` can be a NumPy array or the `WindowView` returned by the helpers below. `fit` converts every batch to float32.

## Starting from a table (most common)

The helpers accept a Polars or a Pandas DataFrame.

1. **One row = one time step.** Columns are:
   - an optional sequence id (for example `seq_id`), when the table holds several independent sequences;
   - an optional time or step column;
   - numeric feature columns.
2. **Order the rows.** Sort them by time inside each sequence, and keep each sequence's rows together (e.g. sort by `[seq_id, step]`). The helpers trust the row order. They don't sort, and they start a new sequence wherever the id value changes.
3. **Pick the columns.**
   - `feature_cols` are the model inputs.
   - `target_cols` are what to predict. They default to `feature_cols`, which means "predict the next step of the same features".
   - Leave id, time and flag columns out of both.
4. **Clean.** There must be no NaN or inf. Fill or drop them before windowing: one NaN spoils every window that contains it.
5. **Scale.** Standardise each feature using the mean and standard deviation of the training part only. Then apply those same numbers to validation and test. RNNs train badly on raw prices or volumes.
6. **Split before windowing.**
   - Split by sequence (whole ids go to either train or validation), or by time (validation comes after train).
   - Never split windows randomly. Neighbouring windows overlap by `window - 1` rows, so a random split leaks validation data into training.
7. **Window.**

   ```python
   from rnns_jax import windows_from_frame

   X, y = windows_from_frame(
       df,
       feature_cols=["x1", "x2"],
       target_cols=["x1"],   # optional, defaults to feature_cols
       id_col="seq_id",      # optional, omit for a single sequence
       window=100,
       horizon=1,
   )
   ```

   - Windows never cross from one sequence into the next.
   - Sequences shorter than `window + horizon` rows are skipped.
   - `stride=k` keeps every k-th window, which gives fewer, less overlapping samples.

## Starting from a single array

```python
from rnns_jax import make_windows

X, y = make_windows(series, window=100)          # series: (T, F)
X, y = make_windows(series, window=100, target=other)  # predict other columns: (T, F_out)
```

A 1-D series needs a feature axis: `series[:, None]`.

## Already have windows?

Pass `X` and `y` to `fit` directly. Only the shapes in the table above matter.

## Memory

`make_windows` and `windows_from_frame` return `X` as a `WindowView`. It behaves like an `(N, window, F)` array: it has `.shape`, `len()` and indexing, and `np.asarray(X)` works. But it stores each row only once, and copies just the windows a batch asks for.

Calling `np.asarray(X)` on a large dataset materialises all windows: N × window × F × 4 bytes. Avoid it unless the data is small.

## Choosing `window` and `horizon`

- **`window`** is how far back the model sees. Longer windows train slower and use more memory per batch.
- **`horizon=1`** predicts the next step. **`horizon=k`** predicts the step k rows after the window ends.

## Batching

`fit` batches the data itself.

- Use `batch_size` ≥ 2. With a batch of 1, the model's `.squeeze()` drops the batch dimension.
- During training, the last incomplete batch is dropped.
- `fit` needs at least `batch_size` training samples.

## Checklist

- [ ] Rows are sorted by time, and each sequence's rows are together.
- [ ] No NaN or inf in the feature or target columns.
- [ ] Features are scaled with statistics from the training part only.
- [ ] Data is split by sequence or by time **before** windowing.
- [ ] `X` has shape `(N, window, F_in)` and `y` has shape `(N, F_out)`.
- [ ] The model has `in_ftrs = F_in` and `out_ftrs = F_out`.
