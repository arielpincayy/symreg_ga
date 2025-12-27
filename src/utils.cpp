#include <utils.h>

bool update_fitness_window(float *window, float new_fitness, int windowsize){
    bool full = true;
    for(int i = windowsize - 1; i > 0; i--){
        window[i] = window[i - 1];
        if(window[i] == std::numeric_limits<float>::max()){
            full = false;
        }
    }
    window[0] = new_fitness;

    if(full){
        float avg = 0.0f;
        for(int i = 0; i < windowsize; i++){
            avg += window[i];
        }
        avg /= windowsize;

        if(fabsf(window[0] - avg) < 1e-6f){
            return true;
        }
    }

    return false;
}

std::string build_expression_rec(int node_idx, int n_leaves, OperatorType *ops, int *terminals, float *consts) {
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
        case ABS: return std::string("|") + left_s + "|";
        case EXP: return std::string("exp(") + left_s + ")";
        case LOG: return std::string("log(") + left_s + ")";
        case NOP: return left_s;
        default: return left_s;
    }
}

std::string build_expression(OperatorType *ops, int *terminals, float *consts, int n_leaves) {
    return build_expression_rec(0, n_leaves, ops, terminals, consts);
}