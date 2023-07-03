#!/bin/bash
#USAGE: ./run_all.sh [SMALLER_MATRIX_SIZE] [BIGGEST_MATRIX_SIZE] [GROWING_STEP]

# $1 == smaller matrix size,
# $2 == biggest matrix size,
# $3 == step of the matrix size

rm -rf csv
mkdir csv
echo "Running naive program"
./bin/naive $1 $2 $3
sleep 30
echo "Running improved program"
./bin/improved $1 $2 $3
sleep 30
echo "Running pthread with 1 thread program"
./bin/pthread $1 $2 $3 1
sleep 30
echo "Running pthread with 8 thread program"
./bin/pthread $1 $2 $3 8
sleep 30
echo "Running pthread with 16 thread program"
./bin/pthread $1 $2 $3 16
sleep 30
echo "Running pthread with 32 thread program"
./bin/pthread $1 $2 $3 32
sleep 30
echo "Running pthread with 64 thread program"
./bin/pthread $1 $2 $3 64
sleep 30
echo "Running pthread with 128 thread program"
./bin/pthread $1 $2 $3 72
sleep 30
echo "Running cuda program"
./bin/cuda $1 $2 $3
sleep 30
echo "Running cuda_row with 1 thread program"
./bin/cuda_row $1 $2 $3 1
sleep 30
echo "Running cuda_row with 8 thread program"
./bin/cuda_row $1 $2 $3 8
echo "Running cuda_row with 16 thread program"
./bin/cuda_row $1 $2 $3 16
sleep 30
echo "Running cuda_row with 64 thread program"
./bin/cuda_row $1 $2 $3 64
sleep 30
echo "Running cuda_row with 128 thread program"
./bin/cuda_row $1 $2 $3 128
sleep 30
echo "Running cuda_row with 512 thread program"
./bin/cuda_row $1 $2 $3 512
sleep 30
echo "Running cuda_row with 1024 thread program"
./bin/cuda_row $1 $2 $3 1024
sleep 30
echo "Running cuda_row with 2048 thread program"
./bin/cuda_row $1 $2 $3 2048
sleep 30
echo "Running cuda_row with 4096 thread program"
./bin/cuda_row $1 $2 $3 4096
sleep 30