#ifndef UTILS_H
#define UTILS_H

#include <limits>
#include <cmath>
#include <string>
#include <individual.cuh>
#include <sstream>
#include <iomanip>

bool update_fitness_window(float *window, float new_fitness, int windowsize);
std::string build_expression_rec(int node_idx, int n_leaves, OperatorType *ops, int *terminals, float *consts);
std::string build_expression(OperatorType *ops, int *terminals, float *consts, int n_leaves);

#endif