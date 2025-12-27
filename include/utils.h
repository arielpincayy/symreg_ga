#ifndef UTILS_H
#define UTILS_H

/**
 * @file utils.h
 * @brief Utility functions for convergence detection and expression building.
 * 
 * Provides helper functions for monitoring evolutionary progress and
 * converting binary tree genomes into human-readable mathematical expressions.
 */

#include <limits>
#include <cmath>
#include <string>
#include <individual.cuh>
#include <sstream>
#include <iomanip>

/**
 * @brief Updates the fitness tracking window and detects convergence/stagnation.
 * 
 * Maintains a sliding window of recent best fitness values to detect when
 * the evolution has stagnated (no improvement over multiple generations).
 * 
 * Algorithm:
 * 1. Shifts all window values right by one position
 * 2. Inserts new fitness at window[0]
 * 3. If window is full, computes average fitness
 * 4. Returns true if current fitness differs from average by less than 1e-6
 * 
 * @param window Array storing recent fitness values (modified in-place)
 * @param new_fitness Latest best fitness value to add
 * @param windowsize Size of the sliding window
 * @return true if convergence detected (stagnation), false otherwise
 * 
 * @note Window is initialized with std::numeric_limits<float>::max()
 * @note Convergence threshold: |window[0] - avg| < 1e-6
 * 
 * @code
 * float window[20];
 * for(int i = 0; i < 20; i++) window[i] = std::numeric_limits<float>::max();
 * 
 * if(update_fitness_window(window, best_fitness, 20)){
 *     cout << "Evolution has stagnated" << endl;
 * }
 * @endcode
 */
bool update_fitness_window(float *window, float new_fitness, int windowsize);

/**
 * @brief Recursively builds a string representation of an expression tree node.
 * 
 * Traverses the binary tree in post-order and constructs a human-readable
 * mathematical expression with proper operator precedence and parentheses.
 * 
 * Tree structure:
 * - Internal nodes (idx < n_leaves-1): operators
 * - Leaf nodes (idx >= n_leaves-1): variables or constants
 * - Left child: 2*idx + 1
 * - Right child: 2*idx + 2
 * 
 * @param node_idx Current node index in the tree (0-based, root at 0)
 * @param n_leaves Total number of leaf nodes in the tree
 * @param ops Array of operators for internal nodes
 * @param terminals Array of terminal indices (-1 for constant, >=0 for variable)
 * @param consts Array of constant values (used when terminals[i] == -1)
 * @return std::string Expression representing the subtree rooted at node_idx
 * 
 * @note Leaf indexing: leaf_idx = node_idx - (n_leaves - 1)
 * @note Variables are formatted as "x0", "x1", "x2", etc.
 * @note Constants are formatted with 3 decimal places
 * 
 * @see build_expression
 */
std::string build_expression_rec(int node_idx, int n_leaves, OperatorType *ops, int *terminals, float *consts);

/**
 * @brief Builds a complete mathematical expression from an individual's genome.
 * 
 * Converts the binary tree representation (operators, terminals, constants)
 * into a human-readable infix notation string suitable for display or logging.
 * 
 * Supported operators:
 * - Binary: +, -, *, /, pow(a,b)
 * - Unary: sin(), cos(), |a|, exp(), log()
 * - Special: NOP (returns left child only)
 * 
 * @param ops Array of operators (size: n_leaves - 1)
 * @param terminals Array of terminal types (size: n_leaves)
 *                  -1 = constant, 0+ = variable index
 * @param consts Array of constant values (size: n_leaves)
 * @param n_leaves Number of leaf nodes in the tree
 * @return std::string Complete mathematical expression
 * 
 * @note This is a wrapper that calls build_expression_rec starting at root (node 0)
 * 
 * @code
 * OperatorType ops[] = {ADD, MUL};
 * int terminals[] = {0, -1, 1}; // x0, constant, x1
 * float consts[] = {0.0f, 2.5f, 0.0f};
 * 
 * string expr = build_expression(ops, terminals, consts, 3);
 * // Result: "(x0 + (2.500 * x1))"
 * @endcode
 * 
 * @see build_expression_rec
 */
std::string build_expression(OperatorType *ops, int *terminals, float *consts, int n_leaves);

#endif