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
enum OperatorType { ADD, SUB, MUL, DIV, SIN, COS, POW, LOG };

extern __device__ __constant__ OperatorType d_operators[];
                                  
const int NUM_OPERATORS = 8;
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

private:
    /**
     * @brief Operation function.
     * @param a Number a.
     * @param b Number b.
     * @param op Operation to apply.
     */
    __device__ 
    float fun(float a, float b, OperatorType op);

    static std::string build_expression_rec(int node_idx, int n_leaves, OperatorType *ops, int *terminals, float *consts);

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
     * @brief Build a human-readable expression for an individual.
     * @return Expression as string.
     */

    static std::string build_expression(OperatorType *ops, int *terminals, float *consts, int n_leaves);
};

#endif // INDIVIDUAL_CUH