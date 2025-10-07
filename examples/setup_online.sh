#!/bin/bash
set -a

SYSTEM=
NEKRS_HOME=
VENV_PATH=
NODES=
TIME=
PROJ_ID=
DEPLOYMENT=
CASE_NAME=

function parse_args() {
  if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
    echo "Invalid number of arguments!"
    echo "Usage:"
    echo "  $0 SYSTEM NEKRS_HOME [VENV_PATH] [NODES] [TIME] [PROJ_ID] [DEPLOYMENT]"
    exit 1
  fi

  SYSTEM="${1,,}"
  NEKRS_HOME=$2
  VENV_PATH=${3:-"./_ssim"}
  NODES=${4:-$(cat ${PBS_NODEFILE} | wc -l)}
  TIME=${5:-"00:30"}
  PROJ_ID=${6:-"datascience"}
  DEPLOYMENT=${7:-"colocated"}

  CASE_NAME=$(ls *.par.safe)
  CASE_NAME=${CASE_NAME:0:${#CASE_NAME}-9}
}

function print_args() {
  echo "SYSTEM    : ${SYSTEM}"
  echo "NEKRS_HOME: ${NEKRS_HOME}"
  echo "VENV_PATH : ${VENV_PATH}"
  echo "NODES     : ${NODES}"
  echo "TIME      : ${TIME}"
  echo "PROJ_ID   : ${PROJ_ID}"
  echo "DEPLOYMENT: ${DEPLOYMENT}"
}

# this maybe not necessary as the nrsrun_${SYSTEM} scripts generate
# them.
function load_modules() {
  if [ ${SYSTEM} == "aurora" ]; then
    module load frameworks
    module load git-lfs/3.5.1
  elif [ ${SYSTEM} == "polaris" ]; then
    module use /soft/modulefiles/
    module load conda
    conda activate
    module load spack-pe-base cmake
  fi
}

function build_smartsim() {
  git clone https://github.com/rickybalin/SmartSim.git
  cd SmartSim
  git checkout rollback_aurora
  pip install -e .

  if [ ${SYSTEM} == "aurora" ]; then
    DEVICE="cpu"
  elif [ ${SYSTEM} == "polaris" ]; then
    DEVICE="gpu"
  fi

  export TORCH_CMAKE_PATH=$( python -c 'import torch;print(torch.utils.cmake_prefix_path)' )
  smart build -v --device $DEVICE --torch_dir $TORCH_CMAKE_PATH --no_tf
  smart validate
  cd ..
}

function setup_venv() {
  if [ ! -d ${VENV_PATH} ]; then
    echo -e "\033[35mPython venv \"${VENV_PATH}\" doesn't exist, building one now ...\033[m"
    python -m venv --clear ${VENV_PATH} --system-site-packages
    source ${VENV_PATH}/bin/activate
  
    if [ ${SYSTEM} == "polaris" ]; then
      export CC=cc
      export CXX=CC
      export CUDNN_BASE=/soft/libraries/cudnn/cudnn-12-linux-x64-v9.1.0.70
      export CUDNN_LIBRARY=$CUDNN_BASE/lib/
      export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include/
      export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH
    fi

    build_smartsim

    export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
    export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
    pip install torch_geometric==2.5.3
  else
    echo -e "\033[35mPython venv \"${VENV_PATH}\" already exists, reusing it ... \033[m"
  fi
}

function generate_script() {
  DB_NODES=1
  SIM_NODES=1
  TRAINER_NODES=1
  if [ ${DEPLOYMENT} == "clustered" ]; then
    echo "Running on $NODES total nodes ..."
    echo "Enter the number of database nodes:"
    read DB_NODES
    echo "Enter the number simulation nodes:"
    read SIM_NODES
    echo "and enter the number of trainer nodes:"
    read TRAIN_NODES
  fi

  NEKRS_HOME=${NEKRS_HOME} VENV_PATH=${VENV_PATH}/bin/activate PROJ_ID=${PROJ_ID} \
  DEPLOYMENT=$DEPLOYMENT DB_NODES=$DB_NODES SIM_NODES=$SIM_NODES TRAIN_NODES=$TRAIN_NODES \
    ./nrsrun_${SYSTEM} ${CASE_NAME} ${NODES} ${TIME}
}

parse_args "$@"
print_args
load_modules
setup_venv
generate_script
