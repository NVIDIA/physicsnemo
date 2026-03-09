<!-- markdownlint-disable -->
# Configuration Layout

## Start here: experiment configs

**Each YAML file in `conf/` is a self-contained experiment config.** Run training or inference by selecting one:

```bash
python train.py --config-name=bumper_geotransolver_oneshot
python train.py --config-name=crash_geotransolver_oneshot
python inference.py --config-name=crash_geotransolver_oneshot
```

To add a new experiment, copy an existing file in `conf/` and edit data paths, model, and features.

---

## Component configs (advanced)

The subfolders (`model/`, `datapipe/`, `reader/`, `training/`, `inference/`) contain configs referenced by experiments. You rarely need to edit them unless customizing models, readers, or training defaults.

| Path           | Purpose                                      |
|----------------|----------------------------------------------|
| `model/`       | Model architectures (selected via experiment) |
| `datapipe/`    | Dataset and feature configs                  |
| `reader/`      | Data format readers (VTP, Zarr)               |
| `training/`    | Training hyperparameters                      |
| `inference/`   | Inference options                            |
