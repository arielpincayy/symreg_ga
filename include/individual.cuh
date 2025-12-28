#ifndef INDIVIDUAL_CUH
#define INDIVIDUAL_CUH

#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <curand_kernel.h>

using namespace std;

/**
 * @brief Types of operators
 */
enum OperatorType { ADD, SUB, MUL, DIV, SIN, COS, ABS, POW, LOG, EXP, NOP };

extern __device__ __constant__ OperatorType d_operators[];

#define MAX_VALUES 256
                                  
const int NUM_OPERATORS = 9;
const float MIN_CONST = -1.0f;
const float MAX_CONST = 1.0f;

/**
 * @brief Stores an individual genetic structure.
 */
struct Genome {
    OperatorType *op;
    int *terminals;
    float *constants;
};

/**
 * @brief Represents an individual.
 * Contains an arbol expression as Genoma.
 */
class Individual {
public:
    float fitness;
    int length;
    int height;
    int nvars;
    int n_leaves;
    Genome genome;

    /**
     * @brief Constructor which creates a random individual.
     * @param h Tree height.
     * @param n_vars Variables number (ej. 1 for 'X', 2 for 'X' and 'Y').
     */
    __device__ 
    Individual(int len, int nleaves, int h, int n_vars, curandState *localState, OperatorType *poolOP, int *poolTerminals, float *poolConsts);
    
    __device__
    Individual(int len, int nleaves, int h, int n_vars, OperatorType *poolOP, int *poolTerminals, float *poolConsts);

private:
    /**
     * @brief Operation function.
     * @param a Number a.
     * @param b Number b.
     * @param op Operation to apply.
     */
    __device__ 
    float fun(float a, float b, OperatorType op);

public:
    /**
     * @brief Evaluate the expression tree.
     * @param input_vars A vector with the input values ​​(e.g. {value of X}).
     * @return Result evaluation (float).
     */
    __device__ 
    float evaluate_tree(const float *input_vars);

    /**
     * @brief Mutate and individual.
     * @param n_mutate Num of mutations.
     * @return New mutated individual.
     */
    __device__
    void mutate(int n_mutate, curandState *localState);

    /**
     * @brief Deep copy constructor.
     * @param other Individual to copy.
     * @return Copy of the original individual.
     */
    __device__ 
    void clone_from(const Individual &parent, OperatorType *my_op, int *my_terminals, float *my_constants);


    /**
     * @brief 
     * 
     * @param A Individual A.
     * @param B Individual B.
     * @param poolOp Pool Operation memory.
     * @param poolTerminals Pool Terminals Memory.
     * @param poolConsts Pool Constants Memory.
     * @return Individual
     */
    __device__
    static Individual crossover(Individual *A, Individual *B, curandState *localState, OperatorType *poolOp, int *poolTerminals, float *poolConsts);
};

#endif // INDIVIDUAL_CUH