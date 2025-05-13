
""" 
Bash script to download the meshgraphnet dataset from deepmind's repo.
      -  Repo: https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
      -  Run: sh download_dataset.sh cylinder_flow
"""

git clone https://github.com/deepmind/deepmind-research.git
set -e
DATASET_NAME="${1}"
OUTPUT_DIR="${DATASET_NAME}"
sh deepmind-research/meshgraphnets/download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
