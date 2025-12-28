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
    int sizey = 4096;          // 10,000 muestras
    int sizeX = sizey * n_vars;

    // 1. MEMORIA DINÁMICA: Para 10,000 muestras es mejor usar malloc
    // El stack de la CPU suele ser limitado y podría dar un "Stack Overflow"
    float *X = (float*)malloc(sizeX * sizeof(float));
    float *y = (float*)malloc(sizey * sizeof(float));

    // Test values. 15 samples x 3 variables (rows):
    /*float X[] = {
        0.1f, 0.2f, 0.3f,
        0.5f, 1.0f, 2.0f,
        1.5f, 2.0f, 3.0f,
        2.5f, 3.0f, 4.0f,
        0.0f, -1.0f, 0.5f,
        3.0f, 2.0f, 1.0f,
        4.0f, 0.5f, -0.5f
    };*/

    int base;
    
    for (int i = 0; i < sizey; i++) {
        base = i * n_vars;
        // Generamos un rango de -2.5 a 2.5 para x0 y x1
        X[base + 0] = -5.0f + (10.0f * i / sizey); 
        X[base + 1] = -5.0f + (10.0f * i / (sizey/2)); // Diferente frecuencia para y
        
        // Inicializar variables extra si n_vars > 2
        for(int v = 2; v < n_vars; v++) X[base + v] = 0.0f;
    }

    // 3. GENERACIÓN DE LA ETIQUETA Y (Ecuación objetivo)
    for (int i = 0; i < sizey; i++) {
        base = i * n_vars;
        float x0 = X[base + 0];
        float x1 = X[base + 1];

        /* * Nota: 1 / (1 + x^-4) es matemáticamente igual a: x^4 / (x^4 + 1)
         * Usamos esta forma porque es más estable si x0 o x1 valen 0.
         */
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

    Operation best = genetic_sym(X, y, sizeX, sizey, n_generations, n_individuals, height, n_vars, tournament_size, n_childs, mut_rate, random_rate, 
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