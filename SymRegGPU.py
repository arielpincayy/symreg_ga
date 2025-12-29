import ctypes
import numpy as np
import os

class Solution(ctypes.Structure):
    _fields_ = [
        ("expression", ctypes.c_char_p),
        ("fitness", ctypes.c_float),
        ("ops", ctypes.POINTER(ctypes.c_int)),       
        ("terminals", ctypes.POINTER(ctypes.c_int)),  
        ("constants", ctypes.POINTER(ctypes.c_float)), 
        ("n_leaves", ctypes.c_int)
    ]

class CUDASymbolicRegressor:
    def __init__(self, lib_path="./bin/libgasymreg.so"):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"No se encontró la librería en {lib_path}")
        
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        
        self.lib.run_genetic_algorithm.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # X
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # y
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # cdf
            ctypes.c_int,   # n_gen
            ctypes.c_int,   # n_ind
            ctypes.c_int,   # tourn
            ctypes.c_int,   # height
            ctypes.c_int,   # n_vars
            ctypes.c_float, # mut
            ctypes.c_float, # repro
            ctypes.c_float, # rand
            ctypes.c_int    # sizey
        ]
        
        self.lib.run_genetic_algorithm.restype = Solution
        self.lib.free_solution.argtypes = [Solution]
        
        self.best_model = None

    def fit(self, X, y, cdf, n_gen=100, n_ind=1024, tourn=15, height=6, mut=0.2, repro=0.7, rand=0.1):
        X = np.require(X, dtype=np.float32, requirements=['C', 'A'])
        y = np.require(y, dtype=np.float32, requirements=['C', 'A'])
        cdf = np.require(cdf, dtype=np.float32, requirements=['C', 'A'])

        n_vars = X.shape[1]
        sizey = len(y)

        result = self.lib.run_genetic_algorithm(
            X, y, cdf, 
            int(n_gen), int(n_ind), int(tourn), int(height), int(n_vars), 
            float(mut), float(repro), float(rand), int(sizey)
        )

        n_leaves = result.n_leaves
        n_ops = n_leaves - 1
        
        self.best_model = {
            'expression': result.expression.decode('utf-8'),
            'fitness': result.fitness,
            'ops': np.ctypeslib.as_array(result.ops, shape=(n_ops,)).copy(),
            'terminals': np.ctypeslib.as_array(result.terminals, shape=(n_leaves,)).copy(),
            'constants': np.ctypeslib.as_array(result.constants, shape=(n_leaves,)).copy(),
            'n_leaves': n_leaves
        }

        self.lib.free_solution(result)
        return self.best_model['expression'], self.best_model['fitness']

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("First you must call .fit()")
            
        X = np.require(X, dtype=np.float32, requirements=['C', 'A'])
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_leaves = self.best_model['n_leaves']
        total_nodes = 2 * n_leaves - 1
        
        values = np.zeros((total_nodes, n_samples), dtype=np.float32)

        terminals = self.best_model['terminals']
        constants = self.best_model['constants']
        
        for i in range(n_leaves):
            idx = n_leaves - 1 + i
            t_code = terminals[i]
            if t_code == -1:
                values[idx, :] = constants[i]
            else:
                values[idx, :] = X[:, t_code]

        ops = self.best_model['ops']
        for i in range(n_leaves - 2, -1, -1):
            a, b = values[2*i+1, :], values[2*i+2, :]
            values[i, :] = self._apply_op(ops[i], a, b)

        return values[0, :]

    def _apply_op(self, op_code, a, b):
        if op_code == 0: return a + b
        if op_code == 1: return a - b
        if op_code == 2: return a * b
        if op_code == 3: return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-6)
        if op_code == 4: return np.sin(a)
        if op_code == 5: return np.cos(a)
        if op_code == 6: return np.abs(a)
        if op_code == 7: return np.power(np.abs(a), b)
        if op_code == 8: return np.log(np.abs(a) + 1e-9)
        if op_code == 9: return np.exp(np.clip(a, -20, 20))
        return a