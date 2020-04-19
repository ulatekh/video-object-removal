#!/usr/bin/env bash

export PATH=/usr/local/cuda/bin:$PATH

TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling channelnorm kernels by nvcc..."
rm -f ChannelNorm_kernel.o
rm -rf ../_ext

nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu -x cu -std=c++11 -Xcompiler -fPIC -arch=sm_30 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include

cd ../
python3 build.py
