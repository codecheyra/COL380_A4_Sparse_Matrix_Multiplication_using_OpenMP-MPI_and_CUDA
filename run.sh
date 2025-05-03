#!/bin/bash

# 1) Load modules
module purge
module load compiler/gcc/9.1/openmpi/4.1.2
module load compiler/gcc/7.1.0/compilervars
module load compiler/cuda/10.0/compilervars
# module load compiler/cuda/11.0/compilervars

# 2) Build
make clean
make

# 3) Disable the openib plugin so mpirun -n 1 … doesn’t warn
# export OMPI_MCA_btl="^openib"

# 4) Run on the small test
# mpirun -n 4 ./a4 small_test
# mpirun -n 2 ./a4 very_small
# mpirun -n 4 ./a4 large_test
mpirun -n 2 ./a4 medium_test_case/medium_test

# 5) Capture the output
# head matrix
