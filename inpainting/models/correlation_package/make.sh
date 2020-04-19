#!/usr/bin/env bash

export PATH=/usr/local/cuda/bin:$PATH

TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src

echo "Compiling correlation kernels by nvcc..."

rm -f correlation_cuda_kernel.o
rm -rf ../_ext

nvcc -c -o correlation_cuda_kernel.o correlation_cuda_kernel.cu -x cu -std=c++11 -Xcompiler -fPIC -arch=sm_30

cd ../
python3 build.py
