#!/usr/bin/env bash
# ---------------------------------------------------------------
# Pull UFS GEFS-v13 replay observation from NOAA S3
# to local lustre storage.
#
# Usage:
#   ./pull_from_noaa_s3.sh              # real copy
#   ./pull_from_noaa_s3.sh --dry-run    # preview only
# ---------------------------------------------------------------
set -euo pipefail

# Set your destination directory
DST_DIR="${HEALDA_RAW_OBS_DIR:-/path/to/your/raw_obs}"


export S5CMD_STAT_PERIOD=10s

YEARS=(2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023)
# Example file structure:
# + **/diag_amsua_*_ges.2018*_control.nc4
# + **/diag_atms_*_ges.2018*_control.nc4
# + **/diag_iasi_*_ges.2018*_control.nc4
# + **/diag_mhs_*_ges.2018*_control.nc4
# + **/diag_cris-fsr_*_ges.2018*_control.nc4
# + **/diag_conv_uv_ges.2018*_control.nc4

# SENSORS=(conv_uv conv_t conv_q conv_gps conv_ps) # Conventional
SENSORS=(amsua amsub atms mhs) # Microwave, can add iasi/cris-fsr for infrared

KIND="ges"          # "ges" or "anl" (contain different innovations but identical "Observation" values)
NUM_WORKERS=512     # s5cmd worker pool
# -----------------------------------------------------------------

DRY=""
[[ ${1:-} == "--dry-run" ]] && DRY="--dry-run"

RUNFILE=$(mktemp)
trap 'rm -f "$RUNFILE"' EXIT

# Build s5cmd run-file: one cp line per year × sensor
# Conventional sensors (conv_*) have no satellite platform in filename
# Satellite sensors have platform suffix (e.g., amsua_metop-b, atms_n20)
for yr in "${YEARS[@]}"; do
  for sensor in "${SENSORS[@]}"; do
    if [[ $sensor == conv_* ]]; then
      # Conventional: diag_conv_gps_ges.2022*_control.nc4
      printf -- 'cp --sp "s3://noaa-ufs-gefsv13replay-pds/%s/*/*/gsi/diag_%s_%s.%s*_control.nc4" "%s/%s/"\n' \
            "$yr" "$sensor" "$KIND" "$yr" "$DST_DIR" "$yr" >>"$RUNFILE"
    else
      # Satellite mw/ir: diag_amsua_*_ges.2022*_control.nc4 (wildcard for platform)
      printf -- 'cp --sp "s3://noaa-ufs-gefsv13replay-pds/%s/*/*/gsi/diag_%s_*_%s.%s*_control.nc4" "%s/%s/"\n' \
            "$yr" "$sensor" "$KIND" "$yr" "$DST_DIR" "$yr" >>"$RUNFILE"
    fi
  done
done

echo ">> built $(wc -l <"$RUNFILE") copy commands in $RUNFILE"

echo ">> Generated commands:"
cat "$RUNFILE"
echo ">> End of generated commands"

s5cmd --no-sign-request --stat        \
      $DRY                            \
      --numworkers "$NUM_WORKERS"     \
      run "$RUNFILE"
