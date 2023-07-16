#!/bin/bash
#USAGE: ./run_all.sh [SMALLER_MATRIX_SIZE] [BIGGEST_MATRIX_SIZE] [GROWING_STEP]

# $1 == smaller matrix size,
# $2 == biggest matrix size,
# $3 == step of the matrix size

rm -rf csv
mkdir csv
echo "Running pthread with 16 thread program"
./bin/pthread 10 500 1 16
echo "Running cuda program"
# ./bin/cuda 10 1200 1
./bin/cuda $1 $2 $3


# ./bin/cuda 10 10000 1
# Saved dim: 1287 time: 34.189342 to csv/load-cuda.csv
# Saved dim: 1287 time: 1.142197 to csv/calc-cuda.csv
# Saved dim: 1287 time: 16.320580 to csv/read-cuda.csv
# Saved dim: 1287 time: 51.652119 to csv/total-cuda.csv
# Saved dim: 1287 time: 50.509922 to csv/overhead-cuda.csv
# ./run.sh: line 13:  9567 Killed                  ./bin/cuda 10 2000 1
# real 149.94
# user 166.63
# sys 18.36
# make: *** [makefile:48: all] Error 137


echo "Running shared program"
./bin/shared $1 $2 $3