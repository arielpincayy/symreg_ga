# Genetic Algorithm for Symbolic Regression (CUDA)

A high-performance GPU-accelerated genetic programming system for symbolic regression using NVIDIA CUDA. This implementation evolves mathematical expressions represented as binary trees to fit given datasets.

## Features

- **Full GPU Acceleration**: All genetic operations run on CUDA-enabled GPUs
- **Expression Trees**: Binary tree representation with various mathematical operators
- **Efficient Memory Management**: Pre-allocated memory pools to avoid dynamic allocation overhead
- **Advanced Selection**: Tournament selection with elitism
- **Genetic Operators**: 
  - Crossover (midpoint)
  - Mutation (operators, constants, variables)
  - Immigration (random individuals injection)
- **Convergence Detection**: Fitness window-based stagnation detection
- **Population Tracking**: Optional logging of all individuals per generation

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability 7.5+)
- CUDA Toolkit 11.0 or higher
- C++11 compatible compiler
- CUB library (included in CUDA Toolkit)

## Project Structure
```
project/
├── Makefile
├── main.cu                 # Entry point
├── include/
│   ├── individual.cuh      # Individual class definition
│   ├── ga.cuh             # GA kernels declarations
│   ├── ga_symreg.cuh      # Main algorithm interface
│   └── utils.h            # Utility functions
└── src/
    ├── ga.cu              # GA kernel implementations
    ├── ga_symreg.cu       # Main algorithm orchestrator
    ├── individual.cu      # Individual methods
    └── utils.cpp          # Expression builder & convergence
```

## Compilation

### Adjust GPU Architecture

Edit the `Makefile` and set the appropriate `-arch` flag for your GPU:
```makefile
NVCC_FLAGS = -std=c++11 -arch=sm_75 -O3 -rdc=true
```

Common architectures:
- `sm_60`: Pascal (GTX 10 series)
- `sm_70`: Volta (V100)
- `sm_75`: Turing (RTX 20 series, GTX 16 series)
- `sm_80`: Ampere (A100)
- `sm_86`: Ampere (RTX 30 series)
- `sm_89`: Ada Lovelace (RTX 40 series)

### Build
```bash
make
```

### Clean Build Files
```bash
make clean
```

## Usage

### Command Line Arguments
```bash
./bin/ga_symreg <n_generations> <n_individuals> <tournament_size> <height> <n_vars> <mut_rate> <n_childs> <random_rate> <write_indiv>
```

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `n_generations` | Maximum number of evolutionary iterations | int | 1000 |
| `n_individuals` | Population size | int | 512 |
| `tournament_size` | Competitors per tournament | int | 7 |
| `height` | Maximum tree height | int | 5 |
| `n_vars` | Number of input variables | int | 3 |
| `mut_rate` | Mutation probability [0.0-1.0] | float | 0.3 |
| `n_childs` | Children per generation | int | 400 |
| `random_rate` | Immigration rate [0.0-1.0] | float | 0.1 |
| `write_indiv` | Enable population logging | int | 0 or 1 |

### Run with Default Parameters
```bash
make run
```

### Run with Custom Parameters
```bash
make run-custom ARGS="500 256 5 4 3 0.2 200 0.05 1"
```

### Example
```bash
./bin/ga_symreg 1000 512 7 5 3 0.3 400 0.1 0
```

This runs 1000 generations with:
- Population of 512 individuals
- Tournament size of 7
- Tree height of 5
- 3 input variables (x0, x1, x2)
- 30% mutation rate
- 400 children per generation
- 10% random immigration
- No population logging

## Supported Operators

| Operator | Symbol | Description |
|----------|--------|-------------|
| `ADD` | + | Addition |
| `SUB` | - | Subtraction |
| `MUL` | * | Multiplication |
| `DIV` | / | Protected division (div by 0 = 1.0) |
| `POW` | ^ | Power (protected for negative bases) |
| `SIN` | sin() | Sine |
| `COS` | cos() | Cosine |
| `ABS` | \| \| | Absolute value |
| `EXP` | exp() | Exponential |
| `LOG` | log() | Natural logarithm (protected) |

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

## Output

### Console Output
```
Solution found at generation 342 with fitness 0.00012
Best fitness: 0.00012
Best equation: (sin(x0) + (x1 * x2)) - log((x0 + 1.000))
```

### Population Log (if enabled)

When `write_indiv=1`, creates `individuals.txt`:
```
generation,individual_id,fitness,expression
0,0,1.234,(x0 + x1)
0,1,2.456,(x0 * 0.500)
...
```

## Fitness Function

Root Mean Square Error (RMSE):
```
fitness = sqrt(Σ(predicted - actual)²)
```

Lower fitness values indicate better solutions.

## Convergence Criteria

The algorithm stops when:
1. **Fitness threshold**: Best fitness < 1e-5
2. **Stagnation**: Fitness window shows no improvement (avg difference < 1e-6)
3. **Max generations**: Reached generation limit

## Performance Tips

1. **Population Size**: Larger populations (512-2048) explore better but are slower
2. **Tree Height**: Height 4-6 balances complexity and search space
3. **Mutation Rate**: 0.2-0.4 works well for most problems
4. **Immigration Rate**: 0.05-0.15 prevents premature convergence
5. **Tournament Size**: 5-10 provides good selection pressure

## Example Problem

The default `main.cu` solves:

**Target Function**: `sin(x0) + (x1 * x2) - log(x0 + 1)`

**Dataset**: 7 samples with 3 variables each

## Customization

### Modify Target Function

Edit `main.cu`:
```cpp
// Generate target values
for(int i = 0; i < sizey; i++){
    int base = i * 3;
    float x0 = X[base + 0];
    float x1 = X[base + 1];
    float x2 = X[base + 2];
    
    // Your target function here
    y[i] = your_function(x0, x1, x2);
}
```

### Add New Operators

1. Add to enum in `individual.cuh`:
```cpp
enum OperatorType { ADD, SUB, ..., YOUR_OP };
```

2. Add to device constant in `individual.cu`:
```cpp
__device__ __constant__ OperatorType d_operators[] = { ADD, SUB, ..., YOUR_OP };
```

3. Implement in `Individual::fun()`:
```cpp
case YOUR_OP: return your_implementation(a, b);
```

4. Update `build_expression_rec()` in `utils.cpp`

## Limitations

- Maximum tree height limited by `MAX_VALUES=256`
- Constant range: [-1.0, 1.0] (adjustable via `MIN_CONST`/`MAX_CONST`)
- No automatic simplification of redundant expressions
- Fixed crossover point (midpoint)

## Known Issues

1. **Bloat**: Large trees with redundant operations may evolve
2. **Numerical instability**: EXP and POW can produce very large values
3. **Premature convergence**: May occur with low immigration rates

## Future Improvements

- [ ] Subtree crossover instead of fixed midpoint
- [ ] Expression simplification
- [ ] Adaptive mutation rates
- [ ] Multi-objective optimization (accuracy + simplicity)
- [ ] Parsimony pressure to control bloat

## License

This project is open source. Feel free to modify and distribute.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- NVIDIA CUB library for efficient GPU sorting
- CUDA toolkit for GPU computing infrastructure