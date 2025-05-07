#!/bin/bash
set -a

if [ -z ${CC} ] || [ -z ${CXX}  ] || [ -z ${FC}  ]; then
  echo -e "\033[31mWARNING: env-var CC, CXX or FC is not set!\033[m"
  export CC=mpicc
  export CXX=mpic++
  export FC=mpif77
  read -p         "Press ENTER to continue with CC=$CC CXX=$CXX FC=$FC or ctrl-c to cancel"
fi

if [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR=$PWD/build
fi
if [ -d ${BUILD_DIR} ]; then
  #rm -r ${BUILD_DIR}
  echo -e "\n\033[31mWARNING! Found a build directory already, you may want to clean it.\033[m\n"
fi

if [[ -z "$INSTALL_DIR" ]]; then
  INSTALL_DIR=$HOME/.local/nekrs
fi
if [ -d ${INSTALL_DIR} ]; then
  #rm -r ${INSTALL_DIR}
  echo -e "\n\033[31mWARNING! Found an install directory already, you may want to clean it.\033[m\n"
fi

cmake -S . -B ${BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}  -Wfatal-errors && \
cmake --build ${BUILD_DIR} --parallel 8 && \
cmake --install ${BUILD_DIR}
if [ $? -eq 0 ]; then
  echo ""
  echo -e "\033[35mHooray! You're all set. The installation is complete.\033[m"
  echo ""
fi
