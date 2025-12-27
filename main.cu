#include <ga_symreg.cuh>


int main(int argc, char **argv) {

    if(argc < 10){
        cerr << "Use: " << argv[0] << " <n_generation> <n_individuals> <tournament_size> <height> <n_vars> <mut_rate> <n_childs> <random_rate> <write_indiv>" << endl;
        return 1;
    }

    int n_generations = atoi(argv[1]);
    int n_individuals = atoi(argv[2]);
    int tournament_size = atoi(argv[3]);
    int height = atoi(argv[4]);
    int n_vars = atoi(argv[5]);
    float mut_rate = atof(argv[6]);
    int n_childs = atoi(argv[7]);
    float random_rate = atof(argv[8]);
    bool write_indiv = atoi(argv[9]);

    int windowsize = 20;
    int sizeX = 21;
    int sizey = 7;

    // Test values. 15 samples x 3 variables (rows):
    float X[] = {
        0.1f, 0.2f, 0.3f,
        0.5f, 1.0f, 2.0f,
        1.5f, 2.0f, 3.0f,
        2.5f, 3.0f, 4.0f,
        0.0f, -1.0f, 0.5f,
        3.0f, 2.0f, 1.0f,
        4.0f, 0.5f, -0.5f
    };
    

    float y[sizey]; 
    for(int i = 0; i < sizey; i++){
        int base = i * 3;
        
        float x0 = X[base + 0];
        float x1 = X[base + 1];
        float x2 = X[base + 2];

        y[i] = sinf(x0) + (x1 * x2) - logf(x0 + 1.0f);
    }
    
    int n_leaves = powf(2, height - 1);
    OperatorType *best_operations = (OperatorType *)malloc((n_leaves - 1) * sizeof(OperatorType));
    int *best_terminals = (int *)malloc(n_leaves * sizeof(int));
    float *best_consts = (float *)malloc(n_leaves * sizeof(float));

    Operation best = genetic_sym(X, y, sizeX, sizey, n_generations, n_individuals, height, n_vars, tournament_size, n_childs, mut_rate, random_rate, 
                                 windowsize, write_indiv, best_operations, best_consts, best_terminals);

    float best_fitness = best.fitness;
    cout << "Best fitness: " << best_fitness << endl;
    std::string expr = build_expression(best.operations, best.terminals, best.consts, n_leaves);
    cout << "Best equation: " << expr << endl;

    free(best_operations);
    free(best_terminals);
    free(best_consts);

    return 0;
}