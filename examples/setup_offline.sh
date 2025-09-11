#!/bin/bash
set -a

SYSTEM=
NEKRS_HOME=
VENV_PATH=
NODES=
TIME=
PROJ_ID=
CASE_NAME=

function parse_args() {
  if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
    echo "Invalid number of arguments!"
    echo "Usage:"
    echo "  $0 SYSTEM NEKRS_HOME [VENV_PATH] [NODES] [TIME] [PROJ_ID]"
    exit 1
  fi

  SYSTEM="${1,,}"
  NEKRS_HOME=$2
  VENV_PATH=${3:-"./_pyg"}
  NODES=${4:-$(cat ${PBS_NODEFILE} | wc -l)}
  TIME=${5:-"00:30"}
  PROJ_ID=${6:-"datascience"}
}

function print_args() {
  echo "SYSTEM    : ${SYSTEM}"
  echo "NEKRS_HOME: ${NEKRS_HOME}"
  echo "VENV_PATH : ${VENV_PATH}"
  echo "NODES     : ${NODES}"
  echo "TIME      : ${TIME}"
  echo "PROJ_ID   : ${PROJ_ID}"

  CASE_NAME=$(ls *.par)
  CASE_NAME=${CASE_NAME:0:${#CASE_NAME}-4}
}

# this maybe not necessary as the nrsrun_${SYSTEM} scripts generate
# them.
function load_modules() {
  if [ ${SYSTEM} == "aurora" ]; then
    module load frameworks
  elif [ ${SYSTEM} == "polaris" ]; then
    module use /soft/modulefiles/
    module load conda
    conda activate
    module load spack-pe-base cmake
  elif [ ${SYSTEM} == "crux" ]; then
    module use /soft/modulefiles/
    module load PrgEnv-gnu/8.5.0
    module load gcc-native/12.3
    module load spack-pe-base/0.8.0
    module load cmake
    module load python/3.10.13
  fi
}

function setup_venv() {
  if [ ! -d ${VENV_PATH} ]; then
    echo -e "\033[35mPython venv \"${VENV_PATH}\" doesn't exist, building one now ...\033[m"
    python -m venv --clear ${VENV_PATH} --system-site-packages
    source ${VENV_PATH}/bin/activate
    if [ ${SYSTEM} == "crux" ]; then
      pip install numpy
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    fi
    TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
    export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
    pip install torch_geometric==2.5.3
  else
    echo -e "\033[35mPython venv \"${VENV_PATH}\" already exists, reusing it ... \033[m"
  fi
}

function generate_script() {
  script=nrsrun_${SYSTEM}

  NEKRS_HOME=${NEKRS_HOME} VENV_PATH=${VENV_PATH}/bin/activate PROJ_ID=${PROJ_ID} ./${script} ${CASE_NAME} ${NODES} ${TIME}
}

parse_args "$@"
print_args
load_modules
setup_venv
generate_script
