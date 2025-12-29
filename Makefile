# --- Configuración del Compilador ---
NVCC = nvcc
# -rdc=true es necesario para que las clases y kernels se vean entre archivos
# -Xcompiler -fPIC es obligatorio para crear la librería .so
NVCC_FLAGS = -std=c++11 -arch=sm_75 -O3 -rdc=true -Xcompiler -fPIC
INCLUDE_DIRS = -I./include

# --- Directorios ---
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# --- Archivos Fuente ---
# Fuentes base (la lógica del algoritmo)
CU_SOURCES = $(SRC_DIR)/ga.cu \
             $(SRC_DIR)/ga_symreg.cu \
             $(SRC_DIR)/individual.cu
CPP_SOURCES = $(SRC_DIR)/utils.cpp

# Fuentes de entrada (Ejecutable vs Wrapper para Python)
MAIN_SOURCE = main.cu
WRAPPER_SOURCE = wrapper.cu

# --- Archivos Objeto ---
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
MAIN_OBJECT = $(OBJ_DIR)/main.o
WRAPPER_OBJECT = $(OBJ_DIR)/wrapper.o

# Objetos comunes que necesitan tanto el EXE como la LIB
COMMON_OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

# --- Objetivos Finales ---
TARGET_EXE = $(BIN_DIR)/ga_symreg
TARGET_LIB = $(BIN_DIR)/libgasymreg.so

# Objetivo por defecto: compila ambos
all: directories $(TARGET_EXE) $(TARGET_LIB)

# Crear carpetas si no existen
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# --- Linker: Ejecutable ---
$(TARGET_EXE): $(MAIN_OBJECT) $(COMMON_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -o $@ $^

# --- Linker: Librería Compartida (.so) ---
$(TARGET_LIB): $(WRAPPER_OBJECT) $(COMMON_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -shared -o $@ $^

# --- Reglas de Compilación ---

# Compilar main.cu
$(MAIN_OBJECT): $(MAIN_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -dc $< -o $@

# Compilar wrapper.cu
$(WRAPPER_OBJECT): $(WRAPPER_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -dc $< -o $@

# Compilar archivos .cu en src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -dc $< -o $@

# Compilar archivos .cpp en src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

# --- Utilidades ---
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) individuals.txt

# Ayuda para los parámetros
help:
	@echo "Estructura detectada en: $(shell pwd)"
	@echo "Targets disponibles:"
	@echo "  make all          - Genera el ejecutable y la librería .so"
	@echo "  make clean        - Borra binarios y objetos"
	@echo ""
	@echo "Para usar con Python, el archivo es: $(TARGET_LIB)"

.PHONY: all clean directories help