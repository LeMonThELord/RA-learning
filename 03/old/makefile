CC        := clang++
LD        := clang++
NVCC	  := nvcc

CPP_FLAG  := -Wall
CPP_DEBUG_FLAG := -g -fsanitize=address
CUDA_FLAG :=
CUDA_DEBUG_FLAG:= -g -G

# MODULES   := widgets test ui
# SRC_DIR   := $(addprefix src/,$(MODULES))
# BUILD_DIR := $(addprefix build/,$(MODULES))

SRC_DIR   := src
BIN_DIR   := bin
BUILD_DIR := build
DEBUG_DIR := debug
DATA_DIR  := csv

CPP_SRC   := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
CPP_OBJ   := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRC))
CPP_DEBUG := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_DIR)/%.o,$(CPP_SRC))
CPP_EXEC  := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%,$(CPP_SRC))

CUDA_SRC  := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))
CUDA_OBJ  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRC))
CPP_DEBUG := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_DIR)/%.o,$(CPP_SRC))
CUDA_EXEC := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(CUDA_SRC))

INCLUDES  := $(addprefix -I,$(SRC_DIR))

vpath %.cpp $(SRC_DIR)
vpath %.cu $(SRC_DIR)

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) -c $^ -o $@
$(BUILD_DIR)/%.o: %.cpp
	$(CC) -c $^ -o $@ $(CPPFLAG)

$(DEBUG_DIR)/%.o: %.cu
	$(NVCC) -c $^ -o $@ $(CUDA_DEBUG_FLAG)
$(DEBUG_DIR)/%.o: %.cpp
	$(LD) -c $^ -o $@ $(CPP_DEBUG_FLAG) $(CPPFLAG)

.phony: all checkdirs clean clean-data debug-build # debug-make

all: checkdirs $(CPP_EXEC) $(CUDA_EXEC) run.sh
	time -p ./run.sh 10 1500 1
	python plot.py

debug-build: checkdirs $(CPP_EXEC) $(CUDA_EXEC) $(CPP_DEBUG) $(CUDA_DEBUG)
	# ./bin/naive 10 10 1
	# ./bin/improved 10 10 1
	# ./bin/pthread 10 10 1
	# ./bin/cuda 10 10 4
	# ./bin/cuda_row 10 10 1


$(CPP_EXEC) $(CUDA_EXEC): $(BIN_DIR) $(CPP_OBJ) $(CUDA_OBJ)
	@$(LD) $(BUILD_DIR)/improved.o -o $(BIN_DIR)/improved
	@$(LD) $(BUILD_DIR)/pthread.o -o $(BIN_DIR)/pthread
	$(NVCC) $(BUILD_DIR)/cuda.o -o $(BIN_DIR)/cuda
	$(NVCC) $(BUILD_DIR)/cuda_row.o -o $(BIN_DIR)/cuda_row
	$(NVCC) $(BUILD_DIR)/cuda_row_old.o -o $(BIN_DIR)/cuda_row_old
	$(NVCC) $(BUILD_DIR)/new.o -o $(BIN_DIR)/new

checkdirs: $(BUILD_DIR) $(DEBUG_DIR) $(BIN_DIR) $(DATA_DIR)
$(BUILD_DIR) $(DEBUG_DIR) $(BIN_DIR) $(DATA_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(DEBUG_DIR)
	@rm -rf $(BUILD_DIR)
	@rm -rf $(DATA_DIR)

debug-make:
	@ echo $(CPP_DEBUG)
	@ echo $(CUDA_DEBUG)