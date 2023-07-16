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

.phony: all checkdirs clean clean-data

all: checkdirs $(CPP_EXEC) $(CUDA_EXEC)
	$(BIN_DIR)/cuda 10 10000 10
	$(BIN_DIR)/shared 10 10000 10
	$(BIN_DIR)/arranged 10 10000 10

$(CPP_EXEC) $(CUDA_EXEC): $(BIN_DIR) $(CPP_OBJ) $(CUDA_OBJ)
	$(NVCC) $(BUILD_DIR)/cuda.o -o $(BIN_DIR)/cuda
	$(NVCC) $(BUILD_DIR)/shared.o -o $(BIN_DIR)/shared
	$(NVCC) $(BUILD_DIR)/arranged.o -o $(BIN_DIR)/arranged

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