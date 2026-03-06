#!/bin/bash

mkdir -p results

python compute_probe_kinematics.py \
  --pred_dir ../predicted_vtps/Run51/test_000 \
  --exact_dir ../exact_vtps/Run51/test_000 \
  --driver_points "70658-70659,70654-70655,70664,70656,70657,70660,70661-70663,70665,70676-70679,70680-70684,70695-70759,70775-70777,70760,70757,70666-70675,70644,70646,70651,70650" \
  --passenger_points "78575-78583,78577,78579,78706-78733,78738-78763,78765-78767,78776-78778,78734-78737,78768-78775" \
  --dt 5e-3 \
  --output_plot results/probe_kinematics.png
python compute_l2_error.py --predicted_parent ../predicted_vtps --exact_parent ../exact_vtps --output_plot results/l2_error.png
python plot_cross_section.py --pred_dir ../predicted_vtps/Run51/test_000/ --exact_dir ../exact_vtps/Run51/test_000/ --output_file results/cross_section.png

