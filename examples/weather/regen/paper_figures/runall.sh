#!/bin/bash

set -e
if ! [[ -d figure_data ]]
then
    echo "figure_data/ not found. Did you extract the tar file here?"
    exit 1
fi

if ! [[ -f figure_data/evenmore_random_val_stations.npy ]]; then
    python3 val_stations.py
fi

set -e
project=/path/to/project
conditionalSamples=$project/inferences/peters_step_64_all.nc
unconditionalSamples=$project/inferences/uncondSample/samples.nc


# run these two in parallel
python3 score_inference.py figure_data/scores/peters_step_64_all &
# truth is misabled here. It is actually "hrrr"
python3 score_inference.py -g truth figure_data/scores/hrrr &
wait

export PYTHONPATH=$(pwd)

python3 figure-07/fig_metrics.py
python3 climate.py
