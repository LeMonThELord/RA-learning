#!/bin/bash
#USAGE: ./run_all.sh [SMALLER_MATRIX_SIZE] [BIGGEST_MATRIX_SIZE] [GROWING_STEP]

# $1 == smaller matrix size,
# $2 == biggest matrix size,
# $3 == step of the matrix size

rm -rf csv
mkdir csv
# echo "Running improved program"
# ./bin/improved $1 $2 $3
echo "Running pthread with 1 thread program"
./bin/pthread $1 $2 $3 16
echo "Running cuda row old program"
./bin/cuda_row_old $1 $2 $3 16
echo "Running cuda row program"
./bin/cuda_row $1 $2 $3 16
echo "Running cuda program"
./bin/cuda $1 $2 $3