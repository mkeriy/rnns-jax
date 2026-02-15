# Wunder Challenge (Market State Forecast)

    [Result: 48 / 696](https://wundernn.io/wunder_challenge/leaderboard)

## Mission

Goal is to predict the next market state vector based on the sequence of states that came before it. Think of it as a sequence modeling problem. You'll be given the market's history up to a certain point, and you need to forecast what happens next.

## How it works

The dataset is a single table in Parquet format, containing multiple independent sequences. Here’s what you need to know.

### The data format

Each row in the table represents a single market state at a specific step in a sequence. The table has **N + 3** columns:

*   `seq_ix`: An ID for the sequence. When this number changes, you're starting a new, completely independent sequence.
*   `step_in_seq`: The step number within a sequence (from 0 to 999).
*   `need_prediction`: A boolean that’s `True` if we need a prediction from you for the *next* step, and `False` otherwise.
*   **N feature columns**: The remaining `N` columns are the anonymized numeric features that describe the market state.

## Training

1. Install requrements
2. Prepare Dataset using eda.ipynb
3. Configure training parametrs, examples in folder /configs
4. Train model 

```bash 
    python training.py --config <path to config>
```