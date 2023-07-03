CLANG=clang++ -Wall -g
NVCC=nvcc -g -G

%.o: %.cpp
	$(CLANG) $^ -o $@

%.o: %.cu
	$(NVCC) $^ -o $@

CPP_SRC=$(wildcard *.cpp)
CPP_OBJ=$(CPP_SRC:.cpp=.o)
CUDA_SRC=$(wildcard *.cu)
CUDA_OBJ=$(CUDA_SRC:.cu=.o)

all: $(CPP_OBJ) $(CUDA_OBJ)
	@ # echo $(CPP_SRC)
	@ # echo $(CPP_OBJ)
	@ # echo $(CUDA_SRC)
	@ # echo $(CUDA_OBJ)

.phony: clear
clear:
	- @ rm $(CPP_OBJ) $(CUDA_OBJ)
	- @ rm -rf temp


nsys: all
	- rm -rf nsys
	- rm -rf temp
	@ mkdir nsys
	@ mkdir temp
	@ - for x in *.o; do \
		echo $$x; \
		nsys profile --stats=true -o ./temp/$$x.temp ./$$x > ./nsys/$$x.report 2> /dev/null; \
	done

time: all
	@ for x in *.o; do echo $$x; sudo time -p ./$$x; echo; done

test:
	rm -rf temp
	@ - mkdir temp
	echo suan
	# nsys profile --stats=true -o ./temp/temp ./implicit_memory.o > ./temp/temp.report
	nvprof -s ./implicit_memory.o
	echo suan