#!/bin/bash
set -a

# user input
SYSTEM=
NEKRS_HOME=
VENV_PATH="../_gnn"
NODES=$(cat ${PBS_NODEFILE} | wc -l)
TIME="01:00"
PROJ_ID="datascience"
DEPLOYMENT="offline"
CLIENT="posix"
ML_TASK=
DB_NODES=1
SIM_NODES=1
TRAIN_NODES=1
# case name is automatically found
CASE_NAME=

function print_help() {
  echo -e "Usage: $0 <SYSTEM> <NEKRS_HOME> [--venv_path | -v <VENV_PATH>]"\
    "[--nodes | -n <NODES>] [--time | -t <TIME>]\n\t"                     \
    "[--proj_id | -p <PROJ_ID>] [--deployment | -d <DEPLOYMENT>]"         \
    "[--ml_task | -m <ML_TASK>] [--client | -c <CLIENT>]\n\t"             \
    "[--db_nodes | -dn <DB_NODES>] [--sim_nodes | -sn <SIM_NODES>]"       \
    "[--train_nodes | -tn <TRAIN_NODES>] [--help | -h]"
}

function parse_args() {
  if [ "$#" -lt 2 ]; then
    print_help
    exit 1
  fi

  SYSTEM="${1,,}"
  shift
  NEKRS_HOME=$1
  shift

  while [ $# -gt 0 ]; do
    key="$1"
    case $key in
      --venv_path| -v)
        VENV_PATH=$2
        shift; shift
        ;;
      --nodes| -n)
        NODES=$2
        shift; shift
        ;;
      --time| -t)
        TIME=$2
        shift; shift
        ;;
      --proj_id| -p)
        PROJ_ID=$2
        shift; shift
        ;;
      --deployment| -d)
        DEPLOYMENT=${2,,}
        shift; shift
        ;;
      --ml_task| -m)
        ML_TASK=${2,,}
        shift; shift
        ;;
      --client| -c)
        CLIENT=${2,,}
        shift; shift
        ;;
      --db_nodes| -dn)
        DB_NODES=$2
        shift; shift
        ;;
      --sim_nodes| -sn)
        SIM_NODES=$2
        shift; shift
        ;;
      --train_nodes| -tn)
        TRAIN_NODES=$2
        shift; shift
        ;;
      --help| -h)
        print_help
        exit 0
        ;;
      *)
        print_help
        exit 1
    esac
  done
}

function check_args() {
  if [ ${SYSTEM} != "aurora" ] && [ ${SYSTEM} != "polaris" ] && [ ${SYSTEM} != "crux" ]; then
    echo "Invalid system name! Supported systems are aurora, polaris and crux."
    exit 1
  fi
  if [ ! -d ${NEKRS_HOME} ]; then
    echo "NEKRS_HOME does not exist! Check path."
    exit 1
  fi
  if [ ${DEPLOYMENT} != "clustered" ] && [ ${DEPLOYMENT} != "colocated" ] && [ ${DEPLOYMENT} != "offline" ]; then
    echo "Invalid deployment type! Supported options are offline, clustered and colocated."
    exit 1
  fi
  if [ -n "${ML_TASK}" ] && [ "${ML_TASK}" != "train" ] && [ "${ML_TASK}" != "inference" ]; then
    echo "Invalid deployment type! Supported options are undefined, train and inference."
    exit 1
  fi
  if [ ${CLIENT} != "smartredis" ] && [ ${CLIENT} != "adios" ] && [ ${CLIENT} != "posix" ]; then
    echo "Invalid client type! Supported options are posix, smartredis and adios."
    exit 1
  fi
  if [ ${DEPLOYMENT} == "colocated" ]; then
    if [ "$SIM_NODES" -ne "$NODES" ] && [ "$TRAIN_NODES" -ne "$NODES" ]; then
      echo "Invalid number of simulation and/or training nodes."
      echo "The number of simulation and training nodes must equal the number of nodes of the job."
      exit 1
    fi
    if [ "$DB_NODES" -ne "1" ]; then
      echo "Invalid number of database nodes."
      echo "With the colocated deployment, the database nodes is set to 1 since it is not sharded."
      exit 1
    fi
  fi
  if [ ${DEPLOYMENT} == "clustered" ]; then
    if [ "$DB_NODES" -eq "2" ]; then
      echo "Invalid number of database nodes."
      echo "SmartSim does not allow clustered databases of 2 nodes, DB_NODES can be 1 or >=3."
      exit 1
    fi
    if [ $((DB_NODES + SIM_NODES + TRAIN_NODES)) -ne "$NODES" ]; then
      echo "Invalid number of database, simulation and training nodes."
      echo "The sum of nodes assigned to each component must equal the total number of nodes of the job."
      exit 1
    fi
  fi
}

function check_connection() {
  if ! curl -Is --max-time 5 https://example.com >/dev/null; then
    echo "Connection test failed — check proxy or network settings." >&2
    exit 1
  fi
}

function setup_case() {
  if [ "$ML_TASK" == "train" ]; then
    # turbChannel_wallModel_ML example
    CASE_NAME=$(ls *_train.par)
    CASE_NAME=${CASE_NAME:0:${#CASE_NAME}-10}
  elif [ "$ML_TASK" == "inference" ]; then
    # turbChannel_wallModel_ML example
    CASE_NAME=$(ls *_inference.par)
    CASE_NAME=${CASE_NAME:0:${#CASE_NAME}-14}
  elif [ "$DEPLOYMENT" == "offline" ]; then
    # offline example
    CASE_NAME=$(ls *.par)
    CASE_NAME=${CASE_NAME:0:${#CASE_NAME}-4}
  elif [ "${DEPLOYMENT}" == "colocated" ] || [ "${DEPLOYMENT}" == "clustered" ]; then
    # online example
    CASE_NAME=$(ls *.par.safe)
    CASE_NAME=${CASE_NAME:0:${#CASE_NAME}-9}
  else
    echo "Error: Unable to find case name!"
    exit 1
  fi

  if [ -n "$ML_TASK" ]; then
    cp ${CASE_NAME}_${ML_TASK}.box ${CASE_NAME}.box
    cp ${CASE_NAME}_${ML_TASK}.udf ${CASE_NAME}.udf
    cp ${CASE_NAME}_${ML_TASK}.usr ${CASE_NAME}.usr
    cp ${CASE_NAME}_${ML_TASK}.par ${CASE_NAME}.par
    cp ${CASE_NAME}_${ML_TASK}.re2 ${CASE_NAME}.re2
    cp ${CASE_NAME}_${ML_TASK}.oudf ${CASE_NAME}.oudf
  fi
}

function print_args() {
  echo "SYSTEM     : ${SYSTEM}"
  echo "NEKRS_HOME : ${NEKRS_HOME}"
  echo "VENV_PATH  : ${VENV_PATH}"
  echo "NODES      : ${NODES}"
  echo "TIME       : ${TIME}"
  echo "PROJ_ID    : ${PROJ_ID}"
  echo "DEPLOYMENT : ${DEPLOYMENT}"
  echo "CLIENT     : ${CLIENT}"
  echo "ML_TASK    : ${ML_TASK}"
  echo "DB_NODES   : ${DB_NODES}"
  echo "SIM_NODES  : ${SIM_NODES}"
  echo "TRAIN_NODES: ${TRAIN_NODES}"
  echo "CASE_NAME  : ${CASE_NAME}"
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
  elif [ ${SYSTEM} == "crux" ]; then
    module use /soft/modulefiles/
    module load PrgEnv-gnu/8.5.0
    module load gcc-native/12.3
    module load spack-pe-base/0.8.0
    module load cmake
    module load python/3.10.13
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

  export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
  export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
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
  
    if [ "${SYSTEM}" == "polaris" ]; then
      export CC=cc
      export CXX=CC
      export CUDNN_BASE=/soft/libraries/cudnn/cudnn-12-linux-x64-v9.1.0.70
      export CUDNN_LIBRARY=$CUDNN_BASE/lib/
      export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include/
      export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH
    elif [ "${SYSTEM}" == "crux" ]; then
      pip install numpy
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    fi

    export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
    export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
    pip install torch_geometric==2.5.3

    if [ "${CLIENT}" == "smartredis" ]; then
      build_smartsim
    fi
  else
    echo -e "\033[35mPython venv \"${VENV_PATH}\" already exists, reusing it ... \033[m"
    if [ "${CLIENT}" == "smartredis" ]; then
      if ! pip list | grep -q smartsim; then
        echo -e "\033[35mRequested the smartredis backend, but SmartSim is not installed \033[m"
        source ${VENV_PATH}/bin/activate
        build_smartsim
      fi
    fi
  fi
}

function generate_script() {
  NEKRS_HOME=${NEKRS_HOME} VENV_PATH=${VENV_PATH}/bin/activate PROJ_ID=${PROJ_ID} \
  DEPLOYMENT=$DEPLOYMENT DB_NODES=$DB_NODES SIM_NODES=$SIM_NODES TRAIN_NODES=$TRAIN_NODES \
  ML_TASK=${ML_TASK} \
    ./nrsrun_${SYSTEM} ${CASE_NAME} ${NODES} ${TIME}
}

parse_args "$@"
check_args
check_connection
setup_case
print_args
load_modules
setup_venv
generate_script
