import numpy as np

def kernel(x1: np.array, x2: np.array, d: int) -> int:
    return poly_kernel(x1, x2, d)

def w(alpha: np.array, train_x: np.array) -> np.array:
    return sum(alpha[i] * train_x[i] for i in range(len(alpha)))

def poly_kernel(x1: np.array, x2: np.array, d: int) -> int:
    return (1 + x1 @ x2) ** d

def decision_function(x, alpha, trainX, k):
    return sum(alpha[i] * kernel(x, trainX[i], k) for i in range(len(alpha)))

def gaussian_kernel(x1: np.array, x2: np.array, sigma: int) -> int:
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))