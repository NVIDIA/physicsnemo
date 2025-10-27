#!/bin/bash
#############################################################
# Author: Clement Etienam (cetienam@nvidia.com)
# Purpose: Build OPM Flow (2025.04/final) with CUDA (manual patch)
#############################################################

set -euo pipefail

echo "🧹 Cleaning up and setting up prerequisites..."
apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:opm/ppa
apt-get update -y

# Essential build tools and version control
apt-get install -y build-essential cmake gfortran pkg-config git-core

# Documentation tools
apt-get install -y doxygen ghostscript \
  texlive-latex-recommended gnuplot

# MPI + BLAS + Boost + others
# apt-get install -y mpi-default-dev libblas-dev libsuitesparse-dev \
  # libboost-all-dev libtrilinos-zoltan-dev libfmt-dev libcjson-dev

apt-get install -y libsuitesparse-dev \
  libboost-all-dev libtrilinos-zoltan-dev libfmt-dev libcjson-dev
  
# DUNE core modules
apt-get install -y libdune-common-dev \
  libdune-geometry-dev libdune-istl-dev libdune-grid-dev


