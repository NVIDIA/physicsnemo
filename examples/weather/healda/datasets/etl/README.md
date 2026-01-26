# Observation Data ETL

Pipeline to prepare UFS observation data for training.

## Full Pipeline

```
1. pull_from_noaa_s3.sh    Download raw NC4 from NOAA S3
2. etl_unified.py          Convert NC4 → parquet
3. compute_normalizations  Compute stats from processed parquet
4. Re-run etl_unified.py   Regenerate channel_table with actual normalization stats
```

## Step 1: Download

`pull_from_noaa_s3.sh` downloads GSI diagnostic files from the public NOAA GEFS-v13 replay archive.

- Edit `YEARS`, `SENSORS`, `KIND` in the script
- Requires [`s5cmd`](https://github.com/peak/s5cmd#installation)
- Run with `--dry-run` to preview

## Step 2: Process

`etl_unified.py` converts NC4 files to parquet with a unified schema.

Example:
```bash
python3 etl_unified.py --output-dir /path/to/output --sensor amsua,conv,atms,amsub,mhs --num-workers 32
```

## Normalization Stats

Normalization stats (mean/std per channel) are loaded from CSV files in `preprocessing/normalizations/`.

**If CSVs are missing:** ETL defaults to mean=0, std=1 (with a warning). The observation parquet files are still valid — only the `channel_table.parquet` needs regeneration after computing proper stats.

**To recompute stats for new sensors:**
1. Run ETL to produce parquet (stats will default to 0/1)
2. Run `preprocessing/compute_normalizations.py --data-root /path/to/output`
3. Regenerate channel table: `python3 etl_unified.py --output-dir /path/to/output --channel-table-only`

## Output Structure

```
processed_obs_v6_ges/
├── channel_table.parquet   # Channel metadata (IDs, normalization stats)
├── amsua/
│   └── 20220101/0.parquet
├── conv/
│   └── 20220101/0.parquet
└── ...
```

## Schema

Defined in `combined_schema.py`. One row per observation.

**Common fields** (all sensors):
- `Latitude`, `Longitude` — observation location
- `Absolute_Obs_Time` — timestamp (ns precision)
- `DA_window` — 3-hourly assimilation window (used for row grouping)
- `Global_Channel_ID` — unique ID across all sensors/channels
- `Platform_ID` — satellite platform or conventional type
- `Observation` — the measurement value

**Satellite-specific** (nullable for conv):
- `Sat_Zenith_Angle`, `Sol_Zenith_Angle`, `Scan_Angle`

**Conventional-specific** (nullable for satellite):
- `Pressure`, `Height`, `Observation_Type`

**Channel table** (`channel_table.parquet`):
- `Global_Channel_ID` — joins to observation data
- `min_valid`, `max_valid` — valid range for QC
- `mean`, `stddev` — normalization statistics
- `is_conv` — conventional vs satellite flag

Conventional observations with multiple components (e.g., uv winds) are flattened into separate rows sharing the same location/time but different `Global_Channel_ID`.
