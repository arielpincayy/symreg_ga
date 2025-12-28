#include <ga_symreg.cuh>


int main(int argc, char **argv) {

    if(argc < 10){
        cerr << "Use: " << argv[0] << " <n_generation> <n_individuals> <tournament_size> <height> <n_vars> <mut_rate> <reproduc_rate> <random_rate> <write_indiv>" << endl;
        return 1;
    }

    int n_generations = atoi(argv[1]);
    int n_individuals = atoi(argv[2]);
    int tournament_size = atoi(argv[3]);
    int height = atoi(argv[4]);
    int n_vars = atoi(argv[5]);
    float mut_rate = atof(argv[6]);
    float reproduc_rate = atof(argv[7]);
    float random_rate = atof(argv[8]);
    bool write_indiv = atoi(argv[9]);

    int windowsize = 20;
    int sizey = 4096;
    int sizeX = sizey * n_vars;

    float *X = (float*)malloc(sizeX * sizeof(float));
    float *y = (float*)malloc(sizey * sizeof(float));

    int base;
    
    for (int i = 0; i < sizey; i++) {
        base = i * n_vars;
        X[base + 0] = -5.0f + (10.0f * i / sizey); 
        X[base + 1] = -5.0f + (10.0f * i / (sizey/2));
        
        for(int v = 2; v < n_vars; v++) X[base + v] = 0.0f;
    }

    for (int i = 0; i < sizey; i++) {
        base = i * n_vars;
        float x0 = X[base + 0];
        float x1 = X[base + 1];

        float x0_2 = x0 * x0;
        float x0_4 = x0_2 * x0_2;
        
        float x1_2 = x1 * x1;
        float x1_4 = x1_2 * x1_2;

        y[i] = (x0_4 / (x0_4 + 1.0f)) + (x1_4 / (x1_4 + 1.0f));
    }

    
    int n_leaves = powf(2, height - 1);
    OperatorType *best_operations = (OperatorType *)malloc((n_leaves - 1) * sizeof(OperatorType));
    int *best_terminals = (int *)malloc(n_leaves * sizeof(int));
    float *best_consts = (float *)malloc(n_leaves * sizeof(float));

    Operation best = genetic_sym(X, y, sizeX, sizey, n_generations, n_individuals, height, n_vars, tournament_size, reproduc_rate, mut_rate, random_rate, 
                                 windowsize, write_indiv, best_operations, best_consts, best_terminals);

    float best_fitness = best.fitness;
    cout << "Best fitness: " << best_fitness << endl;
    std::string expr = build_expression(best.operations, best.terminals, best.consts, n_leaves);
    cout << "Best equation: " << expr << endl;

    free(best_operations);
    free(best_terminals);
    free(best_consts);
    free(X);
    free(y);

    return 0;
}