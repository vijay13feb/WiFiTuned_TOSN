## WiFiTuned

## Engagement Assessment of users attending online synchronus meeting. Leveraging WiFi CSI for head gesticulation recognition and meeting audio of meeting context



## Requirements

```
pip install numpy scipy pandas pywavelets tqdm tensorflow scikit-learn keras-tcn
```

`keras-tcn` provides the `tcn` package used by `network_utility.py`.

## How to Use

Steps 1-3 are prerequisites this folder doesn't contain scripts for — they take
raw captured CSI and produce, per activity recording, a sanitized amplitude/phase
CSV (one row per packet/symbol, one column per subcarrier stream, with a trailing
label/timestamp column). That CSV is the input step 4 expects.

## 1-3. Signal preprocessing, H estimation, CSI reconstruction

Not included here. Whatever pipeline you use for these needs to end with one CSV
per capture, laid out as:

```
<preprocessed_dir>/<activity_or_class>/<capture_name>.csv
```

where each row is a time step (packet) and the last column is a label/timestamp
value that gets dropped before Doppler computation.

## 4. Doppler computation

```
python3 CSI_doppler_computation.py \
    --input-dir  <preprocessed_dir> \
    --output-dir <doppler_dir> \
    [--num-symbols 31] [--sliding 1] [--noise-floor -0.5] [--num-subcarriers 15]
```

- `--input-dir`: root folder of per-class subfolders of preprocessed CSI CSVs (see step 1-3).
- `--output-dir`: mirrors the same per-class subfolder layout; each capture becomes
  a pickled Doppler profile array at `<doppler_dir>/<class>/<capture_name>.txt`.
- `--num-symbols`: sliding-window length (in symbols) used for the Doppler FFT.
- `--num-subcarriers`: must match how many subcarriers are packed into each row
  of the input CSVs.

## 5. Build train/val/test windows

```
python3 CSI_doppler_create_dataset_train.py \
    <doppler_dir>/ <subdirs> <sample_lengths> <sliding> <window_length> <stride_length> \
    <labels_activities> <n_tot> <start_with>
```

Positional args, in order:

| arg | meaning | typical value |
|---|---|---|
| `dir` | Doppler-profile directory from step 4, **with a trailing `/`** | `./doppler_traces/` |
| `subdirs` | comma-separated experiment subfolders under `dir` to process | `S1a,S1b,S1c` |
| `sample_lengths` | packets per sample (same as `--num-symbols` in step 4) | `31` |
| `sliding` | accepted but currently unused by the script — pass any int | `1` |
| `window_length` | samples per training window | `340` |
| `stride_length` | stride between windows | `30` |
| `labels_activities` | comma-separated activity labels | `Forward,Looking,Nodding,Shaking` |
| `n_tot` | antennas × spatial streams per sample | `4` |
| `start_with` | accepted but currently unused by the script — pass any string | `x` |

Note: the file-name filter inside the script only picks up files whose name
contains `Forward`, `Looking`, `Nodding`, or `Shaking` — rename captures (or
adjust the filter in the script) if your activity labels differ.

For each subdir this creates `train_antennas_<activities>/`,
`val_antennas_<activities>/`, `test_antennas_<activities>/` (window `.txt`
pickles), plus `labels_/files_/num_windows_*_antennas_<activities>.txt` index
files, all inside `<dir>/<subdir>/`.

## 6. Train a model from scratch

```
python3 CSI_network_fresh.py \
    <doppler_dir>/ <subdirs> <feature_length> <sample_length> <channels> \
    <batch_size> <num_tot> <name_base> <activities> \
    [--model-dir models] [--output-dir outputs] [--epochs 8] [--patience 6]
```

- `dir`/`subdirs`/`activities`/`num_tot` must match what you used in step 5.
- `feature_length`: Doppler-bin count (matches the FFT size in step 4, typically `100`).
- `sample_length`: must match `window_length` from step 5 (e.g. `340`).
- `name_base`: prefix used for the TF dataset cache files it creates alongside the script.
- `--model-dir`: where the checkpoint is saved, as `<name_base>_<epochs>ep.h5` (SavedModel format).
- `--output-dir`: where the evaluation metrics (confusion matrices, accuracy,
  precision/recall/F-score) are pickled, as `training_<activities>_<subdirs>_<name_base>.txt`.

Example:

```
python3 CSI_network_fresh.py ./doppler_traces/ S1a,S1b 100 340 1 32 4 phase1 \
    Forward,Looking,Nodding,Shaking --epochs 8 --patience 6
```

## 7. Continue training / evaluate

Same arguments, plus `--load-model` pointing at a checkpoint saved by step 6:

```
python3 CSI_network_test.py \
    <doppler_dir>/ <subdirs> <feature_length> <sample_length> <channels> \
    <batch_size> <num_tot> <name_base> <activities> \
    --load-model models/phase1_8ep.h5 \
    [--model-dir models] [--output-dir outputs] [--epochs 4] [--patience 3]
```

This loads the given checkpoint (already compiled, so training resumes without
recompiling), fine-tunes it for `--epochs` more epochs, saves the updated
checkpoint as `<name_base>_<epochs>ep.h5`, and writes a fresh metrics file the
same way step 6 does.
