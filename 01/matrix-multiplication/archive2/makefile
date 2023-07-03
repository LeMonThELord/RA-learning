NVCC=nvcc
CC=clang++ -Wall

%.o: %.cu
	$(NVCC) -c $^ -o $@
%.o: %.cpp
	$(CC) -c $^ -o $@

CPP_SRC=$(wildcard *.cpp)
CPP_OBJ=$(CPP_SRC:.cpp=.o)

.PHONY: clear
clear:
	@ rm -f *.o
	@ rm -f ./main

all: $(CPP_OBJ)
	$(CC) $(CPP_OBJ) -o ./main
	./main
	@ make --no-print-directory clear
