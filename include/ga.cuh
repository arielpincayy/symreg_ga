#ifndef GA_CHUH
#define GA_CUH

#include "individual.cuh"

/**
 * @file ga.cuh
 * @brief CUDA kernels for genetic algorithm operations in symbolic regression.
 * 
 * This header declares all GPU kernels used in the evolutionary process,
 * including population initialization, selection, crossover, mutation,
 * evaluation, and sorting operations.
 */

/**
 * @brief Evaluates an individual's fitness using RMSE (Root Mean Square Error).
 * 
 * Computes the fitness by evaluating the individual's expression tree on all
 * samples in the dataset and calculating the square root of the sum of squared errors.
 * 
 * @param population Pointer to the population array in device memory
 * @param sizeX Total size of input data array (samples * variables)
 * @param sizey Number of samples in the dataset
 * @param idx Index of the individual to evaluate
 * @param X Input data array (flattened 2D array: samples x variables)
 * @param y Target values array
 * @return float RMSE fitness value (lower is better)
 * 
 * @note This is a device function, callable only from GPU kernels
 */
__device__ float eval(Individual *population, int sizeX, int sizey, int idx, float *X, float *y);

/**
 * @brief Initializes the population with random individuals and evaluates their fitness.
 * 
 * Each thread creates one random individual by:
 * 1. Initializing its random state with the given seed
 * 2. Generating random operators and terminals
 * 3. Cloning the individual to the output array
 * 4. Evaluating its fitness
 * 
 * @param output Output array to store generated individuals
 * @param population_size Total number of individuals to create
 * @param X Input data for fitness evaluation
 * @param y Target values for fitness evaluation
 * @param states Random number generator states (one per thread)
 * @param poolOP Pre-allocated memory pool for operators
 * @param poolTerminals Pre-allocated memory pool for terminal indices
 * @param poolConsts Pre-allocated memory pool for constants
 * @param height Maximum tree height
 * @param n_vars Number of input variables
 * @param n_ops Number of operator nodes in the tree
 * @param n_leaves Number of leaf nodes in the tree
 * @param length Total nodes in the tree (2^height - 1)
 * @param seed Random seed for curand initialization
 * @param sizeX Total size of X array
 * @param sizey Number of samples
 * 
 * @note Launch configuration: sufficient threads to cover population_size
 */
__global__ void create_population(Individual *output, int population_size, float *X, float *y, curandState *states, OperatorType *poolOP, int *poolTerminals, 
                                  float *poolConsts, float *cdf, int height, int n_vars, int n_ops, int n_leaves, int length, unsigned long long seed, 
                                  int sizeX, int sizey);

/**
 * @brief Reorders the population based on sorted fitness indices.
 * 
 * After fitness-based sorting, this kernel rearranges individuals from their
 * original positions to their new sorted positions (best to worst).
 * 
 * @param src Source population array (unsorted)
 * @param dst Destination population array (will be sorted)
 * @param sorted_indices Array of indices after sorting (from CUB RadixSort)
 * @param poolOP Destination memory pool for operators
 * @param poolTerminals Destination memory pool for terminals
 * @param poolConsts Destination memory pool for constants
 * @param population_size Total number of individuals
 * @param n_ops Number of operators per individual
 * @param n_leaves Number of leaves per individual
 * 
 * @note After this kernel, dst[0] contains the best individual
 */
__global__ void reorder_population(Individual *src, Individual *dst, int *sorted_indices, OperatorType *poolOP, int *poolTerminals, float *poolConsts,
                                   int population_size, int n_ops, int n_leaves);

/**
 * @brief Performs tournament selection to create a mating pool.
 * 
 * Each thread runs one tournament:
 * 1. Randomly selects tournament_size individuals
 * 2. Chooses the one with best (lowest) fitness
 * 3. Clones it to the new population
 * 
 * @param old_pop Current population (sorted by fitness)
 * @param new_pop Output population (selected parents)
 * @param population_size Number of individuals
 * @param tournament_size Number of competitors per tournament
 * @param states Random number generator states
 * @param poolOP Destination memory pool for operators
 * @param poolTerminals Destination memory pool for terminals
 * @param poolConsts Destination memory pool for constants
 * @param n_ops Number of operators per individual
 * @param n_leaves Number of leaves per individual
 * @param n_vars Number of input variables
 * 
 * @note Handles NaN fitness values by treating them as worst (INFINITY)
 */
__global__ void tournament_selection(Individual *old_pop, Individual *new_pop, int population_size, int tournament_size, curandState *states,
                                    OperatorType *poolOP, int *poolTerminals, float *poolConsts, int n_ops, int n_leaves, int n_vars);

/**
 * @brief Performs crossover operation to generate offspring.
 * 
 * Creates children by combining two randomly selected parents:
 * - First n_childs threads perform crossover (midpoint crossover)
 * - Remaining threads clone individuals without crossover
 * 
 * @param population Parent population
 * @param children Output array for offspring
 * @param population_size Number of parents
 * @param states Random number generator states
 * @param reproduc_rate Crossover probability
 * @param poolOP Destination memory pool for operators
 * @param poolTerminals Destination memory pool for terminals
 * @param poolConsts Destination memory pool for constants (typo in original: poolCOnsts)
 * @param n_ops Number of operators per individual
 * @param n_leaves Number of leaves per individual
 * @param n_vars Number of input variables
 * 
 * @note Crossover uses fixed midpoint split
 */
__global__ void crossover(Individual *population, Individual *children, int population_size, curandState *states, float reproduc_rate, OperatorType *poolOP, int *poolTerminals,
                          float *poolConsts, int n_ops, int n_leaves, int n_vars);

/**
 * @brief Applies mutation to individuals in the population.
 * 
 * Each individual has mut_rate probability of being mutated.
 * Mutation can affect:
 * - Operators (33% chance)
 * - Constants (33% chance, Gaussian noise)
 * - Variables (33% chance)
 * 
 * @param population Population to mutate
 * @param population_size Number of individuals
 * @param states Random number generator states
 * @param mut_rate Mutation probability [0.0, 1.0]
 * 
 * @note Number of mutations per individual: random 1-3 if mutation occurs
 */
__global__ void mutation(Individual *population, int population_size, curandState *states, float mut_rate);

/**
 * @brief Evaluates fitness for all individuals in the population.
 * 
 * Each thread evaluates one individual by computing RMSE on the dataset.
 * 
 * @param population Population to evaluate
 * @param population_size Number of individuals
 * @param X Input data array
 * @param y Target values array
 * @param sizeX Total size of X array
 * @param sizey Number of samples
 * @param states Random number generator states (unused but kept for consistency)
 * 
 * @note Updates the fitness field of each Individual in-place
 */
__global__ void evaluation(Individual *population, int population_size, float *X, float *y, int sizeX, int sizey, curandState *states);

/**
 * @brief Extracts fitness values and initializes indices for sorting.
 * 
 * Prepares data for CUB RadixSort by:
 * 1. Extracting fitness values (replacing NaN with INFINITY)
 * 2. Initializing index array [0, 1, 2, ..., population_size-1]
 * 
 * @param population Source population
 * @param fitness_keys Output array for fitness values (sortable keys)
 * @param indices Output array for individual indices
 * @param population_size Number of individuals
 * 
 * @note NaN fitness values are converted to INFINITY to sort them last
 */
__global__ void extract_fitness(Individual *population, float *fitness_keys, int *indices, int population_size);

/**
 * @brief Replaces worst individuals with random immigrants.
 * 
 * Immigration strategy (maintains diversity):
 * 1. Population must be pre-sorted by fitness (best to worst)
 * 2. Only replaces worst (random_rate * population_size) individuals
 * 3. Best individuals (including elite) are protected
 * 4. New random individuals are immediately evaluated
 * 
 * @param population Population to modify (must be sorted)
 * @param population_size Total population size
 * @param random_rate Immigration rate [0.0, 1.0]
 * @param states Random number generator states
 * @param poolOP Memory pool for operators
 * @param poolTerminals Memory pool for terminals
 * @param poolConsts Memory pool for constants
 * @param cdf Array of cumulative probabilities (must end in 1.0).
 * @param height Tree height
 * @param n_vars Number of input variables
 * @param n_ops Number of operators per tree
 * @param n_leaves Number of leaves per tree
 * @param length Total nodes per tree
 * @param sizeX Size of input data array
 * @param sizey Number of samples
 * @param X Input data
 * @param y Target values
 * 
 * @note Implements implicit elitism by protecting top individuals
 * @note Only threads with idx >= start_idx perform replacement
 */
__global__ void fill_random(Individual *population, int population_size, float random_rate, curandState *states, OperatorType *poolOP, int *poolTerminals, 
                            float *poolConsts, float *cdf, int height, int n_vars, int n_ops, int n_leaves, int length, int sizeX, int sizey, float *X, float *y);

#endif