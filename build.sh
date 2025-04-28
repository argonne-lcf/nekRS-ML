#!/bin/bash
set -a

module load cmake

CC=mpicc
CXX=mpic++
FC=mpif77

BUILD_DIR=$PWD/build
INSTALL_DIR=$HOME/.local/nekrs

if [ -d ${BUILD_DIR} ]; then
  rm -r ${BUILD_DIR}
fi

if [ -d ${INSTALL_DIR} ]; then
  rm -r ${INSTALL_DIR}
fi

cmake -S . -B ${BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}  -Wfatal-errors && \
cmake --build ${BUILD_DIR} --parallel 8 && \
cmake --install ${BUILD_DIR}
