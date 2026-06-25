# LaRAN Network Evaluation

This release provides the LaRAN association network, pretrained checkpoints,
and a lightweight evaluation script for network-level validation.

## Files

The code release contains:

- `run_eval.py`: evaluate the released checkpoints.
- `laran_model.py`: LaRAN network definition.
- `laran_data_loader.py`: HDF5 validation data loader.
- `requirements.txt`: Python dependencies.
- `pyproject.toml`: formatting and linting configuration.

Place the pretrained checkpoints, validation tasks, and normalization cache in
the repository root:

```text
LaRAN-Clean.pth
LaRAN-Noise50.pth
Task1.h5
Task2.h5
Task3.h5
precomputed_cache.pth
```

The released checkpoints are:

- `LaRAN-Clean.pth`: checkpoint trained with 0% association-label noise.
- `LaRAN-Noise50.pth`: checkpoint trained with 50% association-label noise.

To keep the release compact, each validation task includes 100 scenes rather
than the full 1000-scene validation set. The printed results are therefore
computed on the released 100-scene subset.

Each checkpoint contains the full training checkpoint; the evaluation script
loads the association-network subset required for LaRAN network-level
validation.

## Installation

```bash
python -m pip install -r LaRAN/requirements.txt
```

## Run evaluation

From the repository root:

```bash
python LaRAN/run_eval.py
```

The script evaluates both checkpoints on all three tasks and prints:

```text
Task | Checkpoint | F1 | Precision | Recall | TSR | AUPRC
```

Progress bars are shown while each checkpoint-task pair is being evaluated.

These metrics describe network-level association performance on the released
100-scene validation subset.

Metric definitions:

- `Precision`, `Recall`, and `F1` are row-wise macro association metrics over
  active track queries.
- `TSR` is the row-level exact-match track success rate over active track
  queries.
- `AUPRC` is the mean row-wise average precision over rows that contain at
  least one positive association label.

## Data format

Each HDF5 task file must contain `scene_XXXXX` groups with:

- `ground_truth`: `[state_fields, max_tracks, frames]`
- `measurements`: `[measurement_fields, max_points, frames]`

The model uses measurement fields 0-3 as input features. Measurement field 7 is
used only to construct evaluation labels and is not passed to the network input.
Ground-truth field 0 stores the target ID, and fields 1-6 store the target
state used by the evaluator.

The cache file must contain `norm_params` with `meas_mean`, `meas_std`,
`state_mean`, and `state_std`.
