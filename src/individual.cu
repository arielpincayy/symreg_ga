#include "individual.cuh"

using namespace std;

__device__ __constant__ OperatorType d_operators[] = { ADD, SUB, MUL, DIV, SIN, COS, ABS, POW, LOG, EXP, NOP };

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
Individual::Individual(int len, int nleaves, int h, int n_vars, OperatorType *poolOP, int *poolTerminals, float *poolConsts){
    height = h;
    nvars = n_vars;
    n_leaves = nleaves;
    length = len;
    height = h;
    fitness = 0.0f;

    genome.op = poolOP;
    genome.constants = poolConsts;
    genome.terminals = poolTerminals;
}



__device__ 
float Individual::fun(float a, float b, OperatorType op){
    switch (op) {
        case ADD: return a + b;
        case SUB: return a - b;
        case MUL: return a * b;
        case POW:
        if (a < 0 && fabsf(b - roundf(b)) > 1e-6) return 0.0f;
        return powf(a, b);
        case DIV:
        if (b == 0.0f) return 1.0f;
        return a / b;
        case SIN: return sinf(a);
        case COS: return cosf(a);
        case ABS: return fabs(a);
        case EXP: return expf(a);
        case LOG:
            if (a < 0.0f) return 0.0f;
        return log(a);
        case NOP: return a;
        default: return a;
    } 
}

__device__ 
float Individual::evaluate_tree(const float *input_vars){

    float values[MAX_VALUES];
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
    float p; int leaf,op;
    float sigma = 0.3f;
    for(int i = 0; i < n_mutate; i++){
        p = curand_uniform(localState);

        leaf = curand(localState) % n_leaves;
        op   = curand(localState) % (n_leaves - 1);

        if (p < 0.33f) {
            genome.op[op] = d_operators[curand(localState) % NUM_OPERATORS];
        }
        else if (p < 0.66f) {
            float noise = curand_normal(localState) * sigma;
            genome.constants[leaf] += noise;
            
            if (genome.constants[leaf] < MIN_CONST) genome.constants[leaf] = MIN_CONST;
            if (genome.constants[leaf] > MAX_CONST) genome.constants[leaf] = MAX_CONST;
            
            genome.terminals[leaf] = -1;
        }
        else {
            genome.terminals[leaf] = curand(localState) % nvars;
        }
    }
}

__device__
Individual Individual::crossover(Individual *A, Individual *B, curandState *localState, OperatorType *poolOp, int *poolTerminals, float *poolConsts) {
    Individual child(A->length, A->n_leaves, A->height, A->nvars, poolOp, poolTerminals, poolConsts);
    
    int n_ops = A->n_leaves - 1;
    int cut_op = curand(localState) % (n_ops + 1);
    int cut_leaf = curand(localState) % (A->n_leaves + 1);

    
    for(int i = 0; i < n_ops; i++){
        child.genome.op[i] = (i < cut_op) ? A->genome.op[i] : B->genome.op[i];
    }

    for(int i = 0; i < A->n_leaves; i++){
        if(i < cut_leaf){
            child.genome.constants[i] = A->genome.constants[i];
            child.genome.terminals[i] = A->genome.terminals[i];
        } else {
            child.genome.constants[i] = B->genome.constants[i];
            child.genome.terminals[i] = B->genome.terminals[i];
        }
    }

    return child;
}

__device__
void Individual::clone_from(const Individual &parent, OperatorType *my_op, int *my_terminals, float *my_constants){
    height = parent.height;
    nvars = parent.nvars;
    length = parent.length;
    fitness = parent.fitness;
    n_leaves = parent.n_leaves;

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
