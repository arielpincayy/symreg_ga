#include <ga_symreg.cuh>

Operation genetic_sym(float *X, float *y, int sizeX, int sizey, int n_generations, int n_individuals, int height, int n_vars, int tournament_size, int n_childs,
                      float mut_rate, float random_rate, int windowsize, bool write_indiv, OperatorType *best_operations, float *best_consts, int *best_terminals){
    int n_leaves = powf(2, height - 1);
    int n_ops = n_leaves - 1;
    int length = powf(2, height) - 1;
    unsigned long long seed = time(NULL);
    int numThreadPerBlock = 256;
    int numBlocks = (n_individuals + numThreadPerBlock - 1) / numThreadPerBlock;

    Individual *h_population = (Individual *)malloc(n_individuals * sizeof(Individual));
    Individual *d_output_B;
    Individual *d_output_A;

    curandState *d_states;

    float *d_X;
    float *d_y;

    OperatorType *d_poolOP_A;
    int *d_poolTerminals_A;
    float *d_poolConsts_A;

    OperatorType *d_poolOP_B;
    int *d_poolTerminals_B;
    float *d_poolConsts_B;

    // Arrays para CUB RadixSort
    float *d_fitness_keys;
    float *d_fitness_keys_out;
    int *d_idxIndividuals;
    int *d_idxIndividuals_out;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float best_fitness, old_fitness[windowsize];

    cudaMalloc((void**)&d_X, sizeX * sizeof(float));
    cudaMalloc((void**)&d_y, sizey * sizeof(float));
    cudaMalloc((void**)&d_output_A, n_individuals * sizeof(Individual));
    cudaMalloc((void**)&d_states, n_individuals * sizeof(curandState));
    cudaMalloc((void**)&d_poolOP_A, n_individuals * n_ops * sizeof(OperatorType));
    cudaMalloc((void**)&d_poolTerminals_A, n_individuals * n_leaves * sizeof(int));
    cudaMalloc((void**)&d_poolConsts_A, n_individuals * n_leaves * sizeof(float));
    cudaMalloc((void**)&d_poolOP_B, n_individuals * n_ops * sizeof(OperatorType));
    cudaMalloc((void**)&d_poolTerminals_B, n_individuals * n_leaves * sizeof(int));
    cudaMalloc((void**)&d_poolConsts_B, n_individuals * n_leaves * sizeof(float));
    cudaMalloc((void**)&d_output_B, n_individuals * sizeof(Individual));
    
    // Allocate for sorting
    cudaMalloc((void**)&d_fitness_keys, n_individuals * sizeof(float));
    cudaMalloc((void**)&d_fitness_keys_out, n_individuals * sizeof(float));
    cudaMalloc((void**)&d_idxIndividuals, n_individuals * sizeof(int));
    cudaMalloc((void**)&d_idxIndividuals_out, n_individuals * sizeof(int));
    cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizey * sizeof(float), cudaMemcpyHostToDevice);

    // Host buffers for writing individuals
    OperatorType *h_poolOP = (OperatorType*)malloc(n_individuals * n_ops * sizeof(OperatorType));
    int *h_poolTerminals = (int*)malloc(n_individuals * n_leaves * sizeof(int));
    float *h_poolConsts = (float*)malloc(n_individuals * n_leaves * sizeof(float));

    // Open output file (overwrite at start)
    std::ofstream out_file("individuals.txt");
    if (!out_file) {
        cerr << "Failed to open individuals.txt for writing" << std::endl;
    }

    
    create_population<<<numBlocks, numThreadPerBlock>>>(d_output_A, n_individuals, d_X, d_y, d_states, d_poolOP_A, d_poolTerminals_A, d_poolConsts_A, 
                                                        height, n_vars, n_ops, n_leaves, length, seed, sizeX, sizey);
    cudaDeviceSynchronize();
    
    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_fitness_keys, d_fitness_keys_out, d_idxIndividuals, d_idxIndividuals_out, n_individuals);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for(int i = 0; i < windowsize; i++) old_fitness[i] = std::numeric_limits<float>::max();

    for (int gen = 0; gen < n_generations; gen++) {

        tournament_selection<<<numBlocks, numThreadPerBlock>>>(
            d_output_A, 
            d_output_B, 
            n_individuals, 
            tournament_size, 
            d_states,
            d_poolOP_B, 
            d_poolTerminals_B, 
            d_poolConsts_B, 
            n_ops, n_leaves, n_vars
        );

        crossover<<<numBlocks, numThreadPerBlock>>>(
            d_output_B, 
            d_output_A, 
            n_individuals, 
            d_states, 
            n_childs,
            d_poolOP_A, 
            d_poolTerminals_A, 
            d_poolConsts_A, 
            n_ops, n_leaves, n_vars
        );
    
        mutation<<<numBlocks, numThreadPerBlock>>>(
            d_output_A, 
            n_individuals, 
            d_states, mut_rate
        );
    
        evaluation<<<numBlocks, numThreadPerBlock>>>(
            d_output_A, 
            n_individuals, 
            d_X, d_y, 
            sizeX, sizey, 
            d_states
        );

        extract_fitness<<<numBlocks, numThreadPerBlock>>>(
            d_output_A, 
            d_fitness_keys, 
            d_idxIndividuals_out, 
            n_individuals
        );
        
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_fitness_keys, d_fitness_keys_out, d_idxIndividuals_out, d_idxIndividuals, n_individuals);
        
        reorder_population<<<numBlocks, numThreadPerBlock>>>(
            d_output_A, 
            d_output_B, 
            d_idxIndividuals,
            d_poolOP_B, 
            d_poolTerminals_B, 
            d_poolConsts_B,
            n_individuals, n_ops, n_leaves
        );

        std::swap(d_output_A, d_output_B);
        std::swap(d_poolOP_A, d_poolOP_B);
        std::swap(d_poolTerminals_A, d_poolTerminals_B);
        std::swap(d_poolConsts_A, d_poolConsts_B);

        fill_random<<<numBlocks, numThreadPerBlock>>>(
            d_output_A, 
            n_individuals, 
            random_rate, 
            d_states, 
            d_poolOP_A, 
            d_poolTerminals_A, 
            d_poolConsts_A, 
            height, n_vars, n_ops, n_leaves, length, 
            sizeX, sizey, d_X, d_y
        );

        // Copy full population and genome pools to host and write to file
        cudaMemcpy(h_population, d_output_A, n_individuals * sizeof(Individual), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_poolOP, d_poolOP_A, n_individuals * n_ops * sizeof(OperatorType), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_poolTerminals, d_poolTerminals_A, n_individuals * n_leaves * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_poolConsts, d_poolConsts_A, n_individuals * n_leaves * sizeof(float), cudaMemcpyDeviceToHost);    
        cudaMemcpy(&best_fitness, d_fitness_keys_out, sizeof(float), cudaMemcpyDeviceToHost);

        if(write_indiv && gen%10 == 0){
            OperatorType *ops; int *terms; float *consts;
            for (int i = 0; i < n_individuals; ++i) {
                ops = h_poolOP + i * n_ops;
                terms = h_poolTerminals + i * n_leaves;
                consts = h_poolConsts + i * n_leaves;
                std::string expr = build_expression(ops, terms, consts, n_leaves);
                out_file << gen << "," << i << "," << h_population[i].fitness << "," << expr << "\n";
            }
            out_file.flush();
        }
        
        bool converged = update_fitness_window(old_fitness, best_fitness, windowsize);

        if(best_fitness < 1e-5f || converged){
            cout << "Solution found at generation " << gen << " with fitness " << best_fitness << endl;
            break;
        }
    }

    memcpy(best_operations, h_poolOP, n_ops * sizeof(OperatorType));
    memcpy(best_terminals, h_poolTerminals, n_leaves * sizeof(int));
    memcpy(best_consts, h_poolConsts, n_leaves * sizeof(float));

    Operation res;
    res.operations = best_operations;
    res.terminals = best_terminals;
    res.consts = best_consts;
    res.fitness = best_fitness;

    // Free host memory
    free(h_poolOP);
    free(h_poolTerminals);
    free(h_poolConsts);
    free(h_population);
    out_file.close();

    // Free device memory
    cudaFree(d_temp_storage);
    cudaFree(d_fitness_keys);
    cudaFree(d_fitness_keys_out);
    cudaFree(d_idxIndividuals);
    cudaFree(d_idxIndividuals_out);
    cudaFree(d_output_B);
    cudaFree(d_output_A);
    cudaFree(d_states);
    cudaFree(d_poolOP_A);
    cudaFree(d_poolTerminals_A);
    cudaFree(d_poolConsts_A);
    cudaFree(d_poolOP_B);
    cudaFree(d_poolTerminals_B);
    cudaFree(d_poolConsts_B);
    cudaFree(d_X);
    cudaFree(d_y);

    return res;
}
