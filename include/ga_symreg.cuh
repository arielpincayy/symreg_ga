#ifndef GA_SYMREG_H
#define GA_SYMREG_H


#include <ga.cuh>
#include <utils.h>
#include <iostream>
#include <cub/cub.cuh>
#include <fstream>
#include <sstream>


/**
 * @brief Container structure for the best individual's genetic information.
 * 
 * This structure encapsulates the complete genome and fitness of the winning
 * individual from the genetic algorithm. It serves as the return type for
 * the main symbolic regression function, allowing easy access to the evolved
 * solution's components.
 * 
 * The structure contains pointers to host memory arrays that describe the
 * binary tree representation of the mathematical expression.
 * 
 * @note All pointer members reference host memory, not device memory
 * @note Memory ownership: Caller must allocate arrays before passing to genetic_sym()
 * @note Array sizes: operations[n_leaves-1], terminals[n_leaves], consts[n_leaves]
 * 
 * @see genetic_sym
 * @see Individual
 * 
 * @code
 * int n_leaves = pow(2, height - 1);
 * OperatorType *ops = (OperatorType*)malloc((n_leaves-1) * sizeof(OperatorType));
 * int *terms = (int*)malloc(n_leaves * sizeof(int));
 * float *consts = (float*)malloc(n_leaves * sizeof(float));
 * 
 * Operation best = genetic_sym(X, y, ..., ops, consts, terms);
 * 
 * cout << "Best fitness: " << best.fitness << endl;
 * string expr = build_expression(best.operations, best.terminals, best.consts, n_leaves);
 * 
 * free(ops); free(terms); free(consts);
 * @endcode
 */
struct Operation{
    OperatorType *operations;
    int *terminals;
    float *consts;
    float fitness;
};

/**
 * @brief Executes the main Symbolic Regression engine using Genetic Programming on GPU.
 * * This function coordinates the full evolutionary cycle: initializes the population in VRAM,
 * executes CUDA kernels for selection, crossover, and mutation, and handles convergence logic.
 *
 * @param X Input data (independent variables).
 * @param y Target values (labels/ground truth).
 * @param sizeX Total size of the X array (samples * variables).
 * @param sizey Total number of samples (rows).
 * @param n_generations Maximum number of evolutionary iterations.
 * @param n_individuals Population size (total number of individuals).
 * @param height Maximum allowed height for the expression trees.
 * @param n_vars Number of input variables (e.g., 3 for x0, x1, x2).
 * @param tournament_size Number of individuals competing in each tournament selection.
 * @param n_childs Number of children generated during the crossover phase.
 * @param mut_rate Mutation probability (value between 0.0 and 1.0).
 * @param random_rate Immigration rate (percentage of new random individuals per generation).
 * @param windowsize Size of the fitness window used for stagnation detection.
 * @param write_indiv Boolean flag to enable/disable logging population to disk.
 * @param best_operations [Out] Host buffer to store the winning individual's operators.
 * @param best_consts [Out] Host buffer to store the winning individual's constants.
 * @param best_terminals [Out] Host buffer to store the winning individual's variables.
 * @return Operation Structure containing the best fitness and pointers to the winner's genes.
 */
Operation genetic_sym(float *X, float *y, int sizeX, int sizey, int n_generations, int n_individuals, int height, int n_vars, int tournament_size, int n_childs,
                      float mut_rate, float random_rate, int windowsize, bool write_indiv, OperatorType *best_operations, float *best_consts, int *best_terminals);



#endif