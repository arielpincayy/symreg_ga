import numpy as np
from SymRegGPU import CUDASymbolicRegressor
import matplotlib.pyplot as plt

def generate_pagie_data(n_samples=400):
    """Genera el dataset de Pagie: f(x,y) = x^4/(x^4+1) + y^4/(y^4+1)"""
    # Usamos una raíz cuadrada para crear una malla cuadrada
    side = int(np.sqrt(n_samples))
    x = np.linspace(-5, 5, side)
    x0, x1 = np.meshgrid(x, x)
    x0 = x0.flatten().astype(np.float32)
    x1 = x1.flatten().astype(np.float32)
    
    y = (x0**4 / (x0**4 + 1)) + (x1**4 / (x1**4 + 1))
    
    X = np.stack([x0, x1], axis=1)
    return X, y

def main():
    # 1. Instanciar el regresor
    # Asegúrate de que la ruta a libgasymreg.so sea correcta
    regressor = CUDASymbolicRegressor()

    # 2. Preparar Datos
    print("📊 Generando datos de Pagie...")
    X_train, y_train = generate_pagie_data(n_samples=100)
    
    # Pesos de los operadores (CDF)
    # ADD, SUB, MUL, DIV, SIN, COS, ABS, POW, LOG, EXP, NOP
    pesos_cdf = np.array([0.2, 0.4, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0, 1.0], dtype=np.float32)

    # 3. Ejecutar Evolución en GPU
    # Configuramos los parámetros. Height=5 o 6 suele ser suficiente para Pagie.
    n_gen = 500
    n_ind = 4096
    
    raw_eq, cuda_fitness = regressor.fit(
        X_train, 
        y_train, 
        pesos_cdf, 
        n_gen=n_gen, 
        n_ind=n_ind, 
        height=6,
        tourn=20
    )

    print("-" * 40)
    print(f"✅ Evolución Terminada")
    print(f"📡 Fitness reportado por GPU: {cuda_fitness:.6f}")
    print(f"📝 Ecuación encontrada: {raw_eq}")
    print("-" * 40)

    # 4. Predicción usando la estructura del árbol (NumPy Vectorizado)
    print("🤖 Evaluando modelo en Python...")
    y_pred = regressor.predict(X_train)

    # 5. Cálculo de métricas
    # Eliminamos posibles NaNs o Infs por si hubo divisiones extremas
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    rmse = np.sqrt(np.mean((y_train - y_pred_clean)**2))
    
    print(f"📊 Error Final (RMSE en Python): {rmse:.6f}")
    
    # Verificación de consistencia
    diff = abs(rmse - cuda_fitness)
    if diff < 0.01:
        print("✅ Verificación exitosa: El motor de Python coincide con la GPU.")
    else:
        print(f"⚠️ Nota: Diferencia de {diff:.6f} entre Python y GPU (posiblemente por protecciones de operadores).")

    # 6. Opcional: Visualización rápida
    # Si quieres ver si la forma de la superficie es correcta
    try:
        side = int(np.sqrt(len(y_train)))
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Original (Pagie)")
        plt.imshow(y_train.reshape(side, side), extent=[-5,5,-5,5])
        
        plt.subplot(1, 2, 2)
        plt.title("Predicción GPU")
        plt.imshow(y_pred_clean.reshape(side, side), extent=[-5,5,-5,5])
        
        plt.show()
    except Exception as e:
        print(f"No se pudo graficar: {e}")

if __name__ == "__main__":
    main()