#ifndef GA_CHUH
#define GA_CUH

#include "individual.cuh"

__device__ float eval(Individual *population, int sizeX, int sizey, int idx, float *X, float *y);

__global__ void create_population(Individual *output, int population_size, float *X, float *y, curandState *states, OperatorType *poolOP, int *poolTerminals, 
                                  float *poolConsts, int height, int n_vars, int n_ops, int n_leaves, int length, unsigned long long seed, 
                                  int sizeX, int sizey);

__global__ void reorder_population(Individual *src, Individual *dst, int *sorted_indices, OperatorType *poolOP, int *poolTerminals, float *poolConsts,
                                   int population_size, int n_ops, int n_leaves);

__global__ void tournament_selection(Individual *old_pop, Individual *new_pop, int population_size, int tournament_size, curandState *states,
                                    OperatorType *poolOP, int *poolTerminals, float *poolConsts, int n_ops, int n_leaves, int n_vars);

__global__ void crossover(Individual *population, Individual *children, int population_size, curandState *states, int n_childs, OperatorType *poolOP, int *poolTerminals,
                          float *poolCOnsts, int n_ops, int n_leaves, int n_vars);

__global__ void mutation(Individual *population, int population_size, curandState *states, float mut_rate);

__global__ void evaluation(Individual *population, int population_size, float *X, float *y, int sizeX, int sizey, curandState *states);

__global__ void extract_fitness(Individual *population, float *fitness_keys, int *indices, int population_size);

__global__ void fill_random(Individual *population, int population_size, float random_rate, curandState *states, OperatorType *poolOP, int *poolTerminals, 
                            float *poolConsts, int height, int n_vars, int n_ops, int n_leaves, int length, int sizeX, int sizey, float *X, float *y);

#endif