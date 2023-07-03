NVCC=nvcc
CC=clang++ -Wall
DEBUG_FLAGS=-g -fsanitize=address
CUDA_DEBUG_FLAGS=-g -G

.PHONY: clear
clear:
	@ rm -f *.o
	@ rm -f ./main

debug: ./zmain.cpp
	$(CC) -lpthread $(DEBUG_FLAGS) ./zmain.cpp -o ./main
	./main
	@ make --no-print-directory clear

perf: ./zmain.cpp
	$(CC) -lpthread ./zmain.cpp -o ./main
	./main
	@ make --no-print-directory clear

large: ./zmainLarge.cu
	$(NVCC) $(CUDA_DEBUG_FLAGS) ./zmainLarge.cu -o ./main
	./main
	@ make --no-print-directory clear

small: ./zmain.cu
	$(NVCC) ./zmain.cu -o ./main
	./main
	@ make --no-print-directory clear

kernel: ./zmainTimeKernel.cu
	$(NVCC) ./zmainTimeKernel.cu -o ./main
	./main
	@ make --no-print-directory clear


# %.o: %.cu
# 	$(NVCC) -c $^ -o $@
# %.o: %.cpp
# 	$(NVCC) -c $^ -o $@

# CPP_SRC=$(wildcard *.cpp)
# CPP_OBJ=$(CPP_SRC:.cpp=.o)
# CUDA_SRC=$(wildcard *.cu)
# CUDA_OBJ=$(CUDA_SRC:.cu=.o)

# all: $(CPP_OBJ) $(CUDA_OBJ)
# 	$(NVCC) $(CPP_OBJ) $(CUDA_OBJ) -o ./main