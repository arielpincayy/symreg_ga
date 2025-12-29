#include "individual.cuh"

__device__ float eval(Individual *population, int sizeX, int sizey, int idx, float *X, float *y){
    float e;
    float vars[30]; //max 30 variables
    int X_vars = sizeX / sizey;
    // Eval section
    float diff = 0.0f;
    int i,j;
    for (i = 0; i < sizey; i++) {
        for (j = 0; j < X_vars; j++) {
            vars[j] = X[i * X_vars + j];
        }
        e = population[idx].evaluate_tree(vars) - y[i];
        diff += e*e;
    }

    float res = sqrtf(diff/sizey);
    if(isnan(res)) return INFINITY;

    return res;
}

__global__ void create_population(Individual *output, int population_size, float *X, float *y, curandState *states, OperatorType *poolOP, 
                                  int *poolTerminals, float *poolConsts, float *cdf, int height, int n_vars, int n_ops, int n_leaves, int length,
                                  unsigned long long seed, int sizeX, int sizey) {

    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(population_size <= idx) return;

    curand_init(seed, idx, 0, &states[idx]);

    Individual ind(length, n_leaves, height, n_vars, &states[idx], &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves], cdf);

    output[idx].clone_from(ind, &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);
    output[idx].fitness = eval(output, sizeX, sizey, idx, X, y);
}

__global__ void extract_fitness(Individual *population, float *fitness_keys, int *indices, int population_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;
    
    float f = population[idx].fitness;
    fitness_keys[idx] = isnan(f) ? INFINITY : f;
    indices[idx] = idx;
}

__global__ void reorder_population(Individual *src, Individual *dst, int *sorted_indices, OperatorType *poolOP, int *poolTerminals, float *poolConsts,
                                   int population_size, int n_ops, int n_leaves) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;
    
    int src_idx = sorted_indices[idx];
    
    // Clone from sorted position to new ordered position
    dst[idx].clone_from(src[src_idx], &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);
}

// Kernel corregido
__global__ void tournament_selection(Individual *old_pop, Individual *new_pop, int population_size, int tournament_size, curandState *states,
                                    OperatorType *poolOP, int *poolTerminals, float *poolConsts, int n_ops, int n_leaves, int n_vars) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    curandState localState = states[idx];

    int best = curand(&localState) % population_size;
    float best_fit = old_pop[best].fitness;
    int c;
    float f;
    
    for (int i = 1; i < tournament_size; i++) {
        c = curand(&localState) % population_size;
        f = old_pop[c].fitness;
        if (f < best_fit || isnan(best_fit)) {
            best_fit = f;
            best = c;
        }
    }

    new_pop[idx].clone_from(old_pop[best], &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);
    states[idx] = localState;
}

__global__ void crossover(Individual *population, Individual *children, int population_size, curandState *states, float reproduc_rate, OperatorType *poolOP, int *poolTerminals, 
                          float *poolConsts, int n_ops, int n_leaves, int n_vars) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= population_size) return;


    curandState localState = states[idx];


    if(curand_uniform(&localState) < reproduc_rate){
        children[idx].clone_from(population[idx], &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);
        states[idx] = localState;
        return;
    }

    
    int parentA_idx = curand(&localState) % population_size;
    int parentB_idx = curand(&localState) % population_size;

    Individual *A = &population[parentA_idx]; 
    Individual *B = &population[parentB_idx];

    children[idx] = Individual::crossover(A, B, &localState, &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);
    
    states[idx] = localState;
}

__global__ void mutation(Individual *population, int population_size, curandState *states, float mut_rate) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;
    
    curandState localState = states[idx];

    int n_mutate;
    if (curand_uniform(&localState) > mut_rate) {
        states[idx] = localState;
        return;
    }

    n_mutate = 1 + (curand(&localState) % 3);
    population[idx].mutate(n_mutate, &localState);
    states[idx] = localState;

}

__global__ void evaluation(Individual *population, int population_size, float *X, float *y, int sizeX, int sizey, curandState *states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;
    population[idx].fitness = eval(population, sizeX, sizey, idx, X, y);

}

__global__ void fill_random(Individual *population, int population_size, float random_rate, curandState *states, OperatorType *poolOP, 
    int *poolTerminals, float *poolConsts, float *cdf, int height, int n_vars, int n_ops, int n_leaves, int length, int sizeX, int sizey, float *X, float *y) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= population_size) return;

    int n_to_replace = (int)(population_size * random_rate);
    int start_idx = population_size - n_to_replace;

    if (idx < start_idx) return;

    curandState localState = states[idx];

    Individual ind(length, n_leaves, height, n_vars, &localState, &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves], cdf);

    population[idx].clone_from(ind, &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);
    population[idx].fitness = eval(population, sizeX, sizey, idx, X, y);
    
    states[idx] = localState;
}