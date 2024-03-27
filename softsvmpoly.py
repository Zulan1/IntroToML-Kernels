import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from utils import kernel


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """
    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = int(trainX.shape[0])
    zeros = spmatrix([], [], [], (m, m))
    gram_matrix = matrix([[kernel(x1, x2, k) for x1 in trainX] for x2 in trainX])
    H = 2 * l * matrix(sparse([[gram_matrix, zeros], [zeros, zeros]])) + 1e-1 * spdiag([1.0] * 2 * m)
    u = matrix([0.0] * m + [1.0/m] * m)
    v = matrix([1.0] * m + [0.0] * m)
    Im = spmatrix(1.0, range(m), range(m))
    Y = matrix(spdiag(list(float(y) for y in trainy)))
    YG = Y * gram_matrix
    A = matrix(sparse([[YG, zeros], [Im, Im]]))
    print(H.size, u.size, A.size, v.size)
    sol = solvers.qp(H, u, -A, -v)
    return np.vstack(np.array(sol['x']))[:m]

    # print(A)
    # np_gram_matrix = np.array(A)
    # print(np_gram_matrix.shape)
    # w,_ = np.linalg.eig(np_gram_matrix)
    # print(len([w_i for w_i in w if w_i < 0]))


def simple_test():
    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"
    
if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
