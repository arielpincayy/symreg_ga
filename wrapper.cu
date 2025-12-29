#include <ga_symreg.cuh>
#include <utils.h>
#include <cstring>
#include <iostream>

extern "C" {
    struct Solution {
        char *expression;
        float fitness;
        OperatorType *ops;        // Array de OperatorType (int)
        int *terminals;  // Array de terminales
        float *constants; // Array de constantes
        int n_leaves;
    };
    
    Solution run_genetic_algorithm(
        float* X, float* y, float* cdf, 
        int n_gen, int n_ind, int tourn, int height, int n_vars, 
        float mut, float repro, float rand, int sizey           
    ) {
        int sizeX = sizey * n_vars;
        Operation best = genetic_sym(X, y, sizeX, sizey, cdf, n_gen, 
                                     n_ind, height, n_vars, tourn, 
                                     repro, mut, rand, 20);

        int n_leaves = (int)powf(2, height - 1);
        std::string expr = build_expression(best.operations, best.terminals, best.consts, n_leaves);
        
        // No liberamos los arrays todavía, los pasamos a Python
        // Pero usamos malloc para que persistan fuera de esta función
        Solution res;
        res.expression = strdup(expr.c_str());
        res.fitness = best.fitness;
        res.ops = best.operations;    // Transferimos la propiedad de la memoria
        res.terminals = best.terminals;
        res.constants = best.consts;
        res.n_leaves = n_leaves;

        return res;
    }

    // Nueva función para liberar la memoria del árbol desde Python
    void free_solution(Solution sol) {
        if (sol.expression) free(sol.expression);
        if (sol.ops) free(sol.ops);
        if (sol.terminals) free(sol.terminals);
        if (sol.constants) free(sol.constants);
    }
}