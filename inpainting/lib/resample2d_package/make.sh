#!/usr/bin/env bash

export PATH=/usr/local/cuda/bin:$PATH

TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling resample2d kernels by nvcc..."
rm -f Resample2d_kernel.o
rm -rf ../_ext

nvcc -c -o Resample2d_kernel.o Resample2d_kernel.cu -x cu -std=c++11 -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO,-fPIC -arch=sm_30 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include

cd ../
python3 build.py
