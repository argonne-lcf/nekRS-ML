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

function check_args() {
  if [ ${SYSTEM} != "aurora" ] && [ ${SYSTEM} != "polaris" ] && [ ${SYSTEM} != "crux" ]; then
    echo "Invalid system name!"
    exit 1
  fi
  if [ ! -d ${NEKRS_HOME} ]; then
    echo "NEKRS_HOME does not exist!"
    exit 1
  fi
}

function check_connection() {
  if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
    echo "Connection to internet is up, continuing..."
  else
    echo "No internet connection"
    exit 1
  fi
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
    module load frameworks/2025.2.0
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
    
    if [ ${SYSTEM} != "polaris" ]; then
      TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
      export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
      pip install torch_geometric==2.5.3
    fi

    if [ ${SYSTEM} == "aurora" ]; then
      git clone https://github.com/rusty1s/pytorch_cluster.git
      cd pytorch_cluster
      # Need to force the install to not link with OpenMP, change lines 53-54 to if False:
      sed -i 's/fopenmp/lgomp/' setup.py
      CXX=$(which dpcpp) python setup.py install
      cd ..
    elif [ ${SYSTEM} == "polaris" ]; then
      pip install torch-cluster
    fi

    pip install pymech
  else
    echo -e "\033[35mPython venv \"${VENV_PATH}\" already exists, reusing it ... \033[m"
  fi
}

function generate_script() {
  NEKRS_HOME=${NEKRS_HOME} VENV_PATH=${VENV_PATH}/bin/activate PROJ_ID=${PROJ_ID} \
    ./nrsrun_${SYSTEM} ${CASE_NAME} ${NODES} ${TIME}
}

parse_args "$@"
check_args
check_connection
print_args
load_modules
setup_venv
generate_script
