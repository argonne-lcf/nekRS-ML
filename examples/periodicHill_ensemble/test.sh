 export NEKRS_HOME=/flare/AdvanceFusionFission/balin/nekRS-ML/exe
 export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
 mpiexec -n 12 --ppn 12 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- $NEKRS_HOME/bin/nekrs --setup periodicHill --backend dpcpp --build-only
 mpiexec -n 12 --ppn 12 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- $NEKRS_HOME/bin/nekrs --setup periodicHill --backend dpcpp 
