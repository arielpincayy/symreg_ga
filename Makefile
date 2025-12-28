# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -std=c++11 -arch=sm_75 -O3 -rdc=true
INCLUDE_DIRS = -I./include

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
MAIN_SOURCE = main.cu

CU_SOURCES = $(SRC_DIR)/ga.cu \
             $(SRC_DIR)/ga_symreg.cu \
             $(SRC_DIR)/individual.cu

CPP_SOURCES = $(SRC_DIR)/utils.cpp

# Object files
MAIN_OBJECT = $(OBJ_DIR)/main.o
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))

ALL_OBJECTS = $(MAIN_OBJECT) $(CU_OBJECTS) $(CPP_OBJECTS)

# Target executable
TARGET = $(BIN_DIR)/ga_symreg

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Link all object files with device link step
$(TARGET): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -o $@ $^

# Compile main.cu (in root)
$(MAIN_OBJECT): $(MAIN_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -dc $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -dc $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) individuals.txt

# Run with default parameters
run: all
	./$(TARGET) 1000 512 7 5 3 0.3 400 0.1 0

# Run with custom parameters
# Usage: make run-custom ARGS="n_gen n_ind tournament height n_vars mut_rate n_childs random_rate write"
run-custom: all
	./$(TARGET) $(ARGS)

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build files"
	@echo "  run          - Build and run with default parameters"
	@echo "  run-custom   - Build and run with custom parameters"
	@echo ""
	@echo "Example custom run:"
	@echo "  make run-custom ARGS=\"500 256 5 4 3 0.2 200 0.05 1\""
	@echo ""
	@echo "Parameters order:"
	@echo "  1. n_generations"
	@echo "  2. n_individuals"
	@echo "  3. tournament_size"
	@echo "  4. height"
	@echo "  5. n_vars"
	@echo "  6. mut_rate"
	@echo "  7. n_childs"
	@echo "  8. random_rate"
	@echo "  9. write_indiv (0 or 1)"

.PHONY: all clean run run-custom help directories