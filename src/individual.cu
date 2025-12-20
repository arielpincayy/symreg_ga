#include "individual.cuh"
#include <sstream>
#include <iomanip>

using namespace std;

__device__ __constant__ OperatorType d_operators[] = { ADD, SUB, MUL, DIV, SIN, COS, POW, LOG };

__device__ 
Individual::Individual(int len, int nleaves, int h, int n_vars, curandState *localState, OperatorType *poolOP, int *poolTerminals, float *poolConsts){
    height = h;
    nvars = n_vars;
    fitness = 0.0f;
    length = len;
    n_leaves = nleaves;
    int n_ops = n_leaves - 1;

    genome.op = poolOP;
    genome.terminals = poolTerminals;
    genome.constants = poolConsts;

    for (int i = 0; i < n_ops; i++) {
        genome.op[i] = d_operators[curand(localState) % NUM_OPERATORS];
    }

    float prob_variable = 0.5f;
    float p, r;

    for (int i = 0; i < n_leaves; i++) {
        
        p = curand_uniform(localState);

        if (p < prob_variable && nvars > 0) {
            genome.terminals[i] = curand(localState) % nvars;
            
        } else {
            genome.terminals[i] = -1; 

            r = curand_uniform(localState);
            genome.constants[i] = MIN_CONST + r * (MAX_CONST - MIN_CONST);
        }
    }
}


__device__ 
float Individual::fun(float a, float b, OperatorType op){
    if (op == ADD) return a + b;
    if (op == SUB) return a - b;
    if (op == MUL) return a * b;
    if (op == POW){
        if (a < 0 && fabsf(b - roundf(b)) > 1e-6) return 0.0f;
        return powf(a, b);
    }
        
    if (op == DIV) {
        if (b == 0.0f) return 1.0f;
        return a / b;
    }
        
    if (op == SIN) return sinf(a);
    if (op == COS) return cosf(a);
    if (op == LOG){
        if(a < 0.0f) return 0.0f;
        return log(a);
    } 
        
    return a;
}

__device__ 
float Individual::evaluate_tree(const float *input_vars){

    float values[128];
    int terminal_code;

    for (int i = 0; i < n_leaves; i++) {
            
        terminal_code = genome.terminals[i];
            
        if (terminal_code == -1) {
            values[n_leaves - 1 + i] = genome.constants[i];
        } else {
            values[n_leaves - 1 + i] = input_vars[terminal_code];
        }
    }

    int left_child, right_child;
    float a, b;
    for (int i = n_leaves - 2; i >= 0; i--) {
        left_child = 2 * i + 1;
        right_child = 2 * i + 2;

        a = values[left_child];
        b = values[right_child];

        values[i] = fun(a, b, genome.op[i]);
    }
        
    return values[0];
}


__device__
void Individual::mutate(int n_mutate, curandState *localState){
    float p, r;
    int leaf;

    for(int i=0; i<n_mutate; i++){
        p = curand_uniform(localState);
        leaf = curand(localState) % n_leaves;
        if(p < 0.3 && leaf < n_leaves - 1){
            genome.op[leaf] = d_operators[curand(localState) % NUM_OPERATORS];
        }else if(p < 0.6){
            r = curand_uniform(localState);
            genome.constants[leaf] += MIN_CONST + r * (MAX_CONST - MIN_CONST);
        }else{
            genome.terminals[leaf] = (curand(localState) % (nvars + 2)) - 1;
        }

    }
}

__device__
void Individual::clone_from(const Individual &parent, OperatorType *my_op, int *my_terminals, float *my_constants){
    height = parent.height;
    nvars = parent.nvars;
    length = parent.length;
    fitness = parent.fitness;
    n_leaves = parent.n_leaves;

    int n_leaves = powf(2, height - 1);
    int n_ops = n_leaves - 1;

    genome.op = my_op;
    genome.terminals = my_terminals;
    genome.constants = my_constants;

    for (int i = 0; i < n_ops; i++) {
        genome.op[i] = parent.genome.op[i];
    }

    for (int i = 0; i < n_leaves; i++) {
        genome.terminals[i] = parent.genome.terminals[i];
        genome.constants[i] = parent.genome.constants[i];
    }
}

std::string Individual::build_expression_rec(int node_idx, int n_leaves, OperatorType *ops, int *terminals, float *consts) {
    if (node_idx >= n_leaves - 1) {
        int leaf = node_idx - (n_leaves - 1);
        if (terminals[leaf] == -1) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << consts[leaf];
            return oss.str();
        } else {
            return std::string("x") + std::to_string(terminals[leaf]);
        }
    }

    int left = 2 * node_idx + 1;
    int right = 2 * node_idx + 2;

    std::string left_s = build_expression_rec(left, n_leaves, ops, terminals, consts);
    std::string right_s = build_expression_rec(right, n_leaves, ops, terminals, consts);

    OperatorType op = ops[node_idx];
    switch (op) {
        case ADD: return "(" + left_s + " + " + right_s + ")";
        case SUB: return "(" + left_s + " - " + right_s + ")";
        case MUL: return "(" + left_s + " * " + right_s + ")";
        case DIV: return "(" + left_s + " / " + right_s + ")";
        case POW: return "pow(" + left_s + "," + right_s + ")";
        case SIN: return std::string("sin(") + left_s + ")";
        case COS: return std::string("cos(") + left_s + ")";
        case LOG: return std::string("log(") + left_s + ")";
        default: return left_s;
    }
}

std::string Individual::build_expression(OperatorType *ops, int *terminals, float *consts, int n_leaves) {
    return build_expression_rec(0, n_leaves, ops, terminals, consts);
}
