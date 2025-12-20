#include "individual.cuh"
#include <iostream>
#include <limits>

__global__ void create_population(Individual *output, int population_size, float *X, float *y, curandState *states, 
                                  OperatorType *poolOP, int *poolTerminals, float *poolConsts, int height, int n_vars, int n_ops, int n_leaves, int length,
                                  unsigned long long seed, int sizeX, int sizey) {

    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(population_size <= idx) return;

    curand_init(seed, idx, 0, &states[idx]);

    Individual ind(length, n_leaves, height, n_vars, &states[idx], &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);

    // Eval section
    float diff = 0.0f;
    int X_vars = sizeX/sizey;
    float e;
    float vars[30]; //max 30 variables
    for(int i = 0; i < sizey; i++){
        for(int j=0; j < X_vars; j++){
            vars[j] = X[i * X_vars + j];
        }
        e = ind.evaluate_tree(vars) - y[i];
        diff += e*e;
    }

    ind.fitness = diff;

    output[idx] = ind;
}


__global__ void tournament_selection(Individual *old_pop, Individual *new_pop, int population_size, int tournament_size, curandState *states,
                                    OperatorType *poolOP, int *poolTerminals, float *poolConsts, int n_ops, int n_leaves, int n_vars) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    curandState localState = states[idx];

    int best_idx = curand(&localState) % population_size;
    float best_fitness = old_pop[best_idx].fitness;

    for (int i = 1; i < tournament_size; i++) {
        int contender_idx = curand(&localState) % population_size;
        float contender_fitness = old_pop[contender_idx].fitness;

        if (contender_fitness < best_fitness || isnan(best_fitness)) {
            best_fitness = contender_fitness;
            best_idx = contender_idx;
        }
    }

    new_pop[idx].clone_from(old_pop[best_idx], &poolOP[idx * n_ops], &poolTerminals[idx * n_leaves], &poolConsts[idx * n_leaves]);

    states[idx] = localState;
}


__global__ void mutation(Individual *population, int population_size, curandState *states, OperatorType *poolOP, int *poolTerminals, 
                         float *poolConsts, int n_ops, int n_leaves, int n_vars, float mut_rate) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    curandState localState = states[idx];
    if (curand_uniform(&localState) < mut_rate) {
        states[idx] = localState;
        return;
    }
    
    int n_mutate = 1 + (curand(&localState) % 3);
    population[idx].mutate(n_mutate, &localState);
    states[idx] = localState;

}

__global__ void evaluation(Individual *population, int population_size, float *X, float *y, int sizeX, int sizey, curandState *states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    Individual ind = population[idx];

    // Eval section
    float diff = 0.0f;
    int X_vars = sizeX / sizey;
    float e;
    float vars[30]; //max 30 variables
    for (int i = 0; i < sizey; i++) {
        for (int j = 0; j < X_vars; j++) {
            vars[j] = X[i * X_vars + j];
        }
        e = ind.evaluate_tree(vars) - y[i];
        diff += e*e;
    }

    population[idx].fitness = diff;

}


int main(int argc, char **argv) {

    if(argc < 4){
        cerr << "Use: " << argv[0] << "<n_generation> <n_individuals> <tournament_size> <height> <n_vars> <mut_rate>" << endl;
    }

    int n_generation = atoi(argv[1]);
    int n_individuals = atoi(argv[2]);
    int tournament_size = atoi(argv[3]);
    int height = atoi(argv[4]);
    int n_vars = atoi(argv[5]);
    float mut_rate = atof(argv[6]);
    int sizeX = 21;
    int sizey = 7;
    // Test values. 7 samples x 3 variables (rows):
    float X_host[] = {
        0.1f, 0.2f, 0.3f,
        0.5f, 1.0f, 2.0f,
        1.5f, 2.0f, 3.0f,
        2.5f, 3.0f, 4.0f,
        0.0f, -1.0f, 0.5f,
        3.0f, 2.0f, 1.0f,
        4.0f, 0.5f, -0.5f
    };
    // y = sin(x0) + x1 * x2 - log(|x0| + 1)
    float y_host[] = {
        0.0645232f,
        2.073960f,
        6.081204f,
        11.345709f,
        -0.5f,
        0.754826f,
        -2.616240f
    };
    int n_leaves = powf(2, height - 1);
    int n_ops = n_leaves - 1;
    int length = powf(2, height) - 1;
    unsigned long long seed = time(NULL);



    Individual *h_population = (Individual *)malloc(n_individuals * sizeof(Individual));
    Individual *d_output;
    curandState *d_states;

    int numThreadPerBlock = 256;
    int numBlocks = (n_individuals + numThreadPerBlock - 1) / numThreadPerBlock;

    float *d_X;
    float *d_y;

    OperatorType *d_poolOP;
    int *d_poolTerminals;
    float *d_poolConsts;
    Individual *d_new_output;

    cudaMalloc((void**)&d_X, sizeX * sizeof(float));
    cudaMalloc((void**)&d_y, sizey * sizeof(float));
    cudaMalloc((void**)&d_output, n_individuals * sizeof(Individual));
    cudaMalloc((void**)&d_states, n_individuals * sizeof(curandState));
    cudaMalloc((void**)&d_poolOP, n_individuals * n_ops * sizeof(OperatorType));
    cudaMalloc((void**)&d_poolTerminals, n_individuals * n_leaves * sizeof(int));
    cudaMalloc((void**)&d_poolConsts, n_individuals * n_leaves * sizeof(float));
    cudaMalloc((void**)&d_new_output, n_individuals * sizeof(Individual));


    cudaMemcpy(d_X, X_host, sizeX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_host, sizey * sizeof(float), cudaMemcpyHostToDevice);
    create_population<<<numBlocks, numThreadPerBlock>>>(d_output, n_individuals, d_X, d_y, d_states, d_poolOP, d_poolTerminals, d_poolConsts, 
                                                        height, n_vars, n_ops, n_leaves, length, seed, sizeX, sizey);
    cudaMemcpy(h_population, d_output, n_individuals * sizeof(Individual), cudaMemcpyDeviceToHost);


    for (int gen = 0; gen < n_generation; gen++) {

        tournament_selection<<<numBlocks, numThreadPerBlock>>>(
            d_output,
            d_new_output,
            n_individuals,
            tournament_size,
            d_states,
            d_poolOP,
            d_poolTerminals,
            d_poolConsts,
            n_ops,
            n_leaves,
            n_vars
        );
    
        mutation<<<numBlocks, numThreadPerBlock>>>(
            d_new_output,
            n_individuals,
            d_states,
            d_poolOP,
            d_poolTerminals,
            d_poolConsts,
            n_ops,
            n_leaves,
            n_vars,
            mut_rate
        );
    
        evaluation<<<numBlocks, numThreadPerBlock>>>(
            d_new_output,
            n_individuals,
            d_X,
            d_y,
            sizeX,
            sizey,
            d_states
        );
    
        Individual *tmp = d_output;
        d_output = d_new_output;
        d_new_output = tmp;
    }

    // Copy final population from device to host and print best fitness

    // Also copy genome pools to host so we can reconstruct the best equation
    OperatorType *h_poolOP = (OperatorType*)malloc(n_individuals * n_ops * sizeof(OperatorType));
    int *h_poolTerminals = (int*)malloc(n_individuals * n_leaves * sizeof(int));
    float *h_poolConsts = (float*)malloc(n_individuals * n_leaves * sizeof(float));

    cudaMemcpy(h_population, d_output, n_individuals * sizeof(Individual), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_poolOP, d_poolOP, n_individuals * n_ops * sizeof(OperatorType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_poolTerminals, d_poolTerminals, n_individuals * n_leaves * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_poolConsts, d_poolConsts, n_individuals * n_leaves * sizeof(float), cudaMemcpyDeviceToHost);

    float best_fitness = std::numeric_limits<float>::infinity();
    int best_idx = -1;
    for (int i = 0; i < n_individuals; i++) {
        float f = h_population[i].fitness;
        if (isnan(f)) continue;
        if (f < best_fitness) {
            best_fitness = f;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        cout << "Best fitness: " << best_fitness << endl;
        OperatorType *ops = h_poolOP + best_idx * n_ops;
        int *terms = h_poolTerminals + best_idx * n_leaves;
        float *consts = h_poolConsts + best_idx * n_leaves;
        std::string expr = Individual::build_expression(ops, terms, consts, n_leaves);
        cout << "Best equation: " << expr << endl;
    } else {
        cout << "No valid best fitness found" << endl;
    }

    free(h_poolOP);
    free(h_poolTerminals);
    free(h_poolConsts);
    free(h_population);

    cudaFree(d_new_output);
    cudaFree(d_output);
    cudaFree(d_states);
    cudaFree(d_poolOP);
    cudaFree(d_poolTerminals);
    cudaFree(d_poolConsts);
    cudaFree(d_X);
    cudaFree(d_y);

    return 0;
}