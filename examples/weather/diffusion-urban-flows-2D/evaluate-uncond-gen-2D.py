# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Statistical evaluation of unconditionally generated 2D flow fields."""

import hydra
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from datasets.uflow2d import get_data_for_evaluation
from helpers.evaluate_helpers import load_predicted_data
from helpers.stats_helpers import StatisticalEvaluation


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate_uflow")
def main(cfg: DictConfig) -> None:
    """Perform statistical evaluation on unconditionally generated 2D flow fields.

    This function loads training data and predicted flow fields, performs
    statistical evaluations including Reynolds stress analysis, power spectral
    density analysis, and joint PDFs, and saves the results as a PDF report.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing paths to training and predicted data,
        evaluation parameters, and checkpoint information.
    """
    # Get important directories/paths
    train_data_path = cfg.dataset.data_path
    pred_data_path = f"{cfg.evaluate.unconditional.dir}/Ep-{cfg.generation.io.inf_ckpt}-stp-{cfg.generation.sampler.num_steps}-uncond-snaps-{cfg.generation.total_images / 1000}k.h5"
    epoch = cfg.evaluate.unconditional.eval_ckpt
    stats_eval_results_dir = cfg.evaluate.unconditional.stats_eval_results_dir
    num = cfg.evaluate.unconditional.num

    # Load Train/Test data ( with mins and maxs)
    # data, x_axis, y_axis, t, u_min, u_max, v_min, v_max = get_data_for_evaluation(data_path=train_data_path, Train=True)

    data, x_axis, y_axis, t = get_data_for_evaluation(
        data_path=train_data_path, Train=True
    )

    # Load predicted data
    pred_data = load_predicted_data(pred_data_path)

    # renormalize predicted data (renorm done in the loaded data directly)
    # pred_data = renormalize(pred_data, u_min= u_min, u_max= u_max, v_min= v_min,  v_max= v_max)

    # check/assert shapes
    assert data.shape[1:] == pred_data.shape[1:], (
        "Image resolution for gtruth and prediction should match"
    )

    # Statistical evaluation of the Model
    output_file_path = stats_eval_results_dir + f"/epoch-{epoch}"
    stats_eval = StatisticalEvaluation(
        gtruth=data,
        pred=pred_data,
        x_axis=x_axis,
        y_axis=y_axis,
        time=t,
        input_data_type="2D",
        data="line-x",
        output_file_path=output_file_path,
    )

    print(f"Saving pdf at {stats_eval_results_dir}")

    pdf = PdfPages(
        f"{output_file_path}/stats-pred-snaps-{cfg.evaluate.unconditional.predicted_snaps}-inst{num}.pdf"
    )
    pdf = stats_eval.main(
        num=num, locations=[0.5, 1, 2, 3, 4], y=0.5, pdf=pdf
    )  # locations = locations along x, where the PSD needs to be calculated
    pdf.close()


if __name__ == "__main__":
    main()
