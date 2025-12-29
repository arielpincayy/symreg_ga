# CUDA Genetic Algorithm for Symbolic Regression

A high-performance GPU-accelerated genetic programming system for symbolic regression using NVIDIA CUDA. This implementation evolves mathematical expressions represented as binary trees to fit given datasets, with both C++ and Python interfaces.

## Features

- **Full GPU Acceleration**: All genetic operations run on CUDA-enabled GPUs
- **Dual Interface**: Use as standalone C++ executable or Python library
- **Expression Trees**: Binary tree representation with various mathematical operators
- **Efficient Memory Management**: Pre-allocated memory pools to avoid dynamic allocation overhead
- **Advanced Selection**: Tournament selection with elitism
- **Genetic Operators**: 
  - Crossover (midpoint)
  - Mutation (operators, constants, variables)
  - Immigration (random individuals injection)
- **Convergence Detection**: Fitness window-based stagnation detection
- **Python Integration**: Scikit-learn style API with `.fit()` and `.predict()`
- **Operator Weighting**: Customizable probability distribution for operators (CDF)

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability 7.5+)
- CUDA Toolkit 11.0 or higher
- C++11 compatible compiler
- CUB library (included in CUDA Toolkit)
- Python 3.7+ (for Python interface)
- NumPy (for Python interface)

## Project Structure
```
project/
├── Makefile
├── main.cu                  # C++ entry point
├── wrapper.cu               # Python/C interface
├── SymRegGPU.py            # Python class wrapper
├── test.py                 # Python usage example
├── include/
│   ├── individual.cuh      # Individual class definition
│   ├── ga.cuh              # GA kernels declarations
│   ├── ga_symreg.cuh       # Main algorithm interface
│   └── utils.h             # Utility functions
└── src/
    ├── ga.cu               # GA kernel implementations
    ├── ga_symreg.cu        # Main algorithm orchestrator
    ├── individual.cu       # Individual methods
    └── utils.cpp           # Expression builder & convergence
```

## Compilation

### Adjust GPU Architecture

Edit the `Makefile` and set the appropriate `-arch` flag for your GPU:
```makefile
NVCC_FLAGS = -std=c++11 -arch=sm_75 -O3 -rdc=true -Xcompiler -fPIC
```

Common architectures:
- `sm_60`: Pascal (GTX 10 series)
- `sm_70`: Volta (V100)
- `sm_75`: Turing (RTX 20 series, GTX 16 series)
- `sm_80`: Ampere (A100)
- `sm_86`: Ampere (RTX 30 series)
- `sm_89`: Ada Lovelace (RTX 40 series)

### Build Both Executable and Library
```bash
make all
```

This creates:
- `./bin/ga_symreg` (C++ executable)
- `./bin/libgasymreg.so` (Python shared library)

### Clean Build Files
```bash
make clean
```

## Usage

### Python Interface (Recommended)
```python
from SymRegGPU import CUDASymbolicRegressor
import numpy as np

# Generate sample data
X = np.random.randn(1000, 2).astype(np.float32)
y = (X[:, 0]**2 + X[:, 1]**2).astype(np.float32)

# Define operator weights (CDF)
# Order: ADD, SUB, MUL, DIV, SIN, COS, ABS, POW, LOG, EXP, NOP
cdf = np.array([0.2, 0.4, 0.7, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.0], dtype=np.float32)

# Create and train model
model = CUDASymbolicRegressor()
expression, fitness = model.fit(
    X, y, cdf,
    n_gen=100,      # generations
    n_ind=512,      # population size
    tourn=7,        # tournament size
    height=5,       # max tree height
    mut=0.3,        # mutation rate
    repro=0.7,      # reproduction rate
    rand=0.1        # immigration rate
)

print(f"Best equation: {expression}")
print(f"Fitness (RMSE): {fitness}")

# Make predictions
y_pred = model.predict(X)
```

#### CDF (Cumulative Distribution Function)

The CDF array controls operator probabilities:
```python
# Example: Favor arithmetic operations, disable transcendental functions
cdf = np.array([
    0.25,  # ADD   (25%)
    0.50,  # SUB   (25%)
    0.75,  # MUL   (25%)
    0.95,  # DIV   (20%)
    0.95,  # SIN   (0%)
    0.95,  # COS   (0%)
    0.95,  # ABS   (0%)
    0.95,  # POW   (0%)
    0.95,  # LOG   (0%)
    0.95,  # EXP   (0%)
    1.00   # NOP   (5%)
], dtype=np.float32)
```

### C++ Interface

#### Command Line Arguments
```bash
./bin/ga_symreg <n_generations> <n_individuals> <tournament_size> <height> <n_vars> <mut_rate> <reproduc_rate> <random_rate> <write_indiv>
```

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `n_generations` | Maximum number of evolutionary iterations | int | 1000 |
| `n_individuals` | Population size | int | 512 |
| `tournament_size` | Competitors per tournament | int | 7 |
| `height` | Maximum tree height | int | 5 |
| `n_vars` | Number of input variables | int | 2 |
| `mut_rate` | Mutation probability [0.0-1.0] | float | 0.3 |
| `reproduc_rate` | Reproduction probability [0.0-1.0] | float | 0.7 |
| `random_rate` | Immigration rate [0.0-1.0] | float | 0.1 |
| `write_indiv` | Enable population logging | int | 0 or 1 |

#### Example
```bash
./bin/ga_symreg 1000 512 7 5 2 0.3 0.7 0.1 0
```

## API Reference

### Python Class: `CUDASymbolicRegressor`

#### Methods

**`__init__()`**
```python
model = CUDASymbolicRegressor()
```
Initializes the regressor and loads the CUDA library.

**`fit(X, y, cdf, n_gen=100, n_ind=1024, tourn=15, height=6, mut=0.2, repro=0.7, rand=0.1)`**
```python
expression, fitness = model.fit(X, y, cdf, ...)
```
Trains the model on dataset `(X, y)`.

**Parameters:**
- `X`: Input features (n_samples, n_features), dtype=float32
- `y`: Target values (n_samples,), dtype=float32
- `cdf`: Operator probabilities (11,), dtype=float32
- `n_gen`: Maximum generations
- `n_ind`: Population size
- `tourn`: Tournament size
- `height`: Maximum tree height
- `mut`: Mutation rate
- `repro`: Reproduction rate (probability of cloning without crossover)
- `rand`: Immigration rate

**Returns:**
- `expression`: String representation of best equation
- `fitness`: RMSE of best solution

**`predict(X)`**
```python
y_pred = model.predict(X)
```
Evaluates the trained model on new data.

**Parameters:**
- `X`: Input features (n_samples, n_features), dtype=float32

**Returns:**
- `y_pred`: Predicted values (n_samples,)

### C Structure: `Solution`
```cpp
struct Solution {
    char *expression;        // String representation
    float fitness;           // RMSE value
    OperatorType *ops;       // Operator nodes
    int *terminals;          // Terminal types (-1=const, ≥0=var)
    float *constants;        // Constant values
    int n_leaves;           // Number of leaf nodes
};
```

## Supported Operators

| Operator | Symbol | Description | Safe Guard |
|----------|--------|-------------|------------|
| `ADD` | + | Addition | None |
| `SUB` | - | Subtraction | None |
| `MUL` | * | Multiplication | None |
| `DIV` | / | Protected division | Returns ∞ if \|b\| < 1e-6 |
| `POW` | pow(a,b) | Power | Uses \|a\|, returns a if invalid |
| `SIN` | sin() | Sine (unary) | None |
| `COS` | cos() | Cosine (unary) | None |
| `ABS` | \|·\| | Absolute value (unary) | None |
| `EXP` | exp() | Exponential (unary) | None |
| `LOG` | log() | Natural log (unary) | Uses \|a\| |
| `NOP` | - | No operation (returns left child) | None |

## Algorithm Flow

1. **Initialization**: Random population generation on GPU
2. **Fitness Evaluation**: RMSE calculation against target data
3. **Sorting**: Population sorted by fitness using CUB RadixSort
4. **Selection**: Tournament selection to create mating pool
5. **Crossover**: Generate offspring via midpoint crossover
6. **Mutation**: Random modifications to operators/constants/variables
7. **Evaluation**: Fitness calculation for new individuals
8. **Immigration**: Replace worst individuals with random ones
9. **Convergence Check**: Monitor fitness window for stagnation
10. **Repeat** until convergence or max generations

## Fitness Function

Root Mean Square Error (RMSE):
```
fitness = sqrt(Σ(predicted - actual)² / n_samples)
```

Lower fitness values indicate better solutions.

## Convergence Criteria

The algorithm stops when:
1. **Fitness threshold**: Best fitness < 1e-5
2. **Stagnation**: Fitness window (size=20) shows no improvement (avg difference < 1e-6)
3. **Max generations**: Reached generation limit

## Example: Pagie Polynomial
```python
import numpy as np
from SymRegGPU import CUDASymbolicRegressor

# Generate Pagie-1 polynomial dataset
# f(x,y) = 1/(1+x^-4) + 1/(1+y^-4)
n = 4096
x = np.linspace(-5, 5, int(np.sqrt(n)))
x0, x1 = np.meshgrid(x, x)
X = np.stack([x0.flatten(), x1.flatten()], axis=1).astype(np.float32)
y = (X[:, 0]**4 / (X[:, 0]**4 + 1) + X[:, 1]**4 / (X[:, 1]**4 + 1)).astype(np.float32)

# Favor arithmetic operators
cdf = np.array([0.2, 0.4, 0.68, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 1.0], dtype=np.float32)

model = CUDASymbolicRegressor()
expr, fit = model.fit(X, y, cdf, n_gen=1000, n_ind=2048, height=7)

print(f"Solution: {expr}")
print(f"RMSE: {fit:.6f}")

# Test predictions
y_pred = model.predict(X)
print(f"R² score: {1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2):.4f}")
```

## Output Examples

### Console Output
```
🚀 Iniciando evolución en GPU para 4096 puntos...
Solution found at generation 84 with fitness 0.734462
------------------------------
✅ Evolución terminada.
Mejor fitness: 0.734462
Mejor ecuación: ((x0 * x0) / ((x0 * x0) + 1.000)) + ((x1 * x1) / ((x1 * x1) + 1.000))
```

### Population Log (C++ only, if `write_indiv=1`)

Creates `individuals.txt`:
```
generation,individual_id,fitness,expression
0,0,1.234,(x0 + x1)
0,1,2.456,(x0 * 0.500)
10,0,0.543,((x0 * x0) + x1)
...
```

## Performance Tips

1. **Population Size**: 512-2048 for complex problems, 256-512 for simple ones
2. **Tree Height**: 
   - Height 4-5: Simple expressions
   - Height 6-7: Moderate complexity
   - Height 8+: Complex but slower
3. **Mutation Rate**: 0.2-0.4 works well for most problems
4. **Reproduction Rate**: 0.6-0.8 balances exploitation and exploration
5. **Immigration Rate**: 0.05-0.15 prevents premature convergence
6. **Tournament Size**: 5-10 provides good selection pressure

## Customization

### Modify Target Function (C++)

Edit `main.cu`:
```cpp
for (int i = 0; i < sizey; i++) {
    int base = i * n_vars;
    float x0 = X[base + 0];
    float x1 = X[base + 1];
    
    // Your target function here
    y[i] = your_function(x0, x1);
}
```

### Adjust Operator Probabilities

The CDF must be:
- Monotonically increasing
- End at 1.0
- Length 11 (one per operator)
```python
# Example: Only basic arithmetic
cdf = np.array([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
```

### Add New Operators

1. Add to enum in `individual.cuh`:
```cpp
enum OperatorType { ADD, SUB, ..., YOUR_OP, NOP };
```

2. Update `NUM_OPERATORS` constant

3. Add to device constant in `individual.cu`:
```cpp
__device__ __constant__ OperatorType d_operators[] = { ADD, ..., YOUR_OP, NOP };
```

4. Implement in `Individual::fun()`:
```cpp
case YOUR_OP: return your_implementation(a, b);
```

5. Update `build_expression_rec()` in `utils.cpp`

6. Update `_apply_op()` in `SymRegGPU.py`

## Limitations

- Maximum tree height limited by `MAX_VALUES=256`
- Maximum 30 input variables (adjustable in `ga.cu`)
- Constant range: [-1.0, 1.0] (adjustable via `MIN_CONST`/`MAX_CONST`)
- No automatic simplification of redundant expressions
- Fixed crossover point (midpoint)

## Known Issues

1. **Bloat**: Large trees with redundant operations may evolve
2. **Numerical instability**: EXP and POW can produce very large values
3. **Premature convergence**: May occur with low immigration rates
4. **Memory leak**: Small leak in wrapper (3 arrays per `.fit()` call)

## Troubleshooting

**Problem**: `libgasymreg.so not found`
```bash
# Ensure library is built
make all
# Check it exists
ls -la bin/libgasymreg.so
```

**Problem**: CUDA out of memory
- Reduce `n_ind` (population size)
- Reduce `height` (tree complexity)
- Check GPU memory: `nvidia-smi`

**Problem**: Invalid pointer / segfault
- Ensure CUDA architecture matches your GPU (`-arch` flag)
- Check array sizes match between C++ and Python

## Future Improvements

- [ ] Subtree crossover instead of fixed midpoint
- [ ] Expression simplification
- [ ] Adaptive mutation rates
- [ ] Multi-objective optimization (accuracy + simplicity)
- [ ] Parsimony pressure to control bloat
- [ ] Support for multiple GPUs
- [ ] Batch prediction in CUDA
- [ ] Save/load trained models

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly (both C++ and Python)
4. Submit a pull request

## License

This project is open source under the MIT License.

## Acknowledgments

- NVIDIA CUB library for efficient GPU sorting
- CUDA toolkit for GPU computing infrastructure
- Inspired by PySR and gplearn symbolic regression libraries

## Citation

If you use this software in your research, please cite:
```
@software{symreg_ga,
  title={CUDA Genetic Algorithm for Symbolic Regression},
  author={Ariel Pincay Pérez},
  year={2025},
  url={https://github.com/arielpincayy/symreg_ga}
}
```