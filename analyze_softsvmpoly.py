import numpy as np
import matplotlib.pyplot as plt
from softsvmpoly import softsvmpoly
from utils import kernel
import time

ROUND_DIGITS = 5

def scatter_plot(title, x_label, y_label):
    """plots a scatter plot with the given parameters"""
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    trainX_blue = trainX[trainy == 1]
    trainX_red = trainX[trainy == -1]
    print(trainX_blue[:10])
    plt.scatter(*zip(*trainX_blue), color='blue', label='y=1')
    plt.scatter(*zip(*trainX_red), color='red', label='y=-1')
    plt.title(title, fontsize=26)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(True)

def predict_calculate_error(alpha, trainX, testX, testy, k):
    """predicts the testy values and calculates the error"""
    def decision_function(x):
        return sum(alpha[i] * kernel(x, trainX[i], k) for i in range(len(alpha)))
    y_preds = np.array([np.sign(decision_function(x)) for x in testX])
    return np.mean(np.vstack(testy) != np.vstack(y_preds))

def test_input_size(m: int = 200, l: int = 1, k: int = 5):
    """tests the ssvm algorithm for a given sample size and l value"""
    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    w = softsvmpoly(l, k, _trainX, _trainy)

    test_error = predict_calculate_error(w, trainX, testX, testy, k)
    train_error = predict_calculate_error(w, trainX, _trainX, _trainy, k)

    return test_error, train_error

def k_fold_cross_validation(k_f: int, l: int, k: int, trainX: np.array, trainy: np.array):
    m, _ = trainX.shape
    indices = np.random.permutation(m)
    fold_size = m // k_f
    trainX_folds = [trainX[indices[i:i + fold_size]] for i in range(0, m, fold_size)]
    trainy_folds = [trainy[indices[i:i + fold_size]] for i in range(0, m, fold_size)]

    test_errors = []
    for i in range(k_f):
        valX = trainX_folds[i]
        valY = trainy_folds[i]
        trainX = np.concatenate([trainX_folds[j] for j in range(k_f) if j != i])
        trainy = np.concatenate([trainy_folds[j] for j in range(k_f) if j != i])
        w = softsvmpoly(l, k, trainX, trainy)
        test_errors.append(predict_calculate_error(w, trainX, valX, valY, k))

    print(f'k={k}, l={l}, test_errors={test_errors}')    
    return np.average(test_errors)

def analyze_lambda_k_values(k_f, l_values, k_values, train_x, train_y, test_x, test_y):
    """tests the knn algorithm for different k"""   
    best_k, best_l, best_error = 0, 0, 1
    for k in k_values:
        for l in l_values:
            test_error = k_fold_cross_validation(k_f, l, k, train_x, train_y)
            best_error, best_k, best_l = min((best_error, best_k, best_l), (test_error, k, l), key=lambda x: x[0])
            print(f'k={k}, l={l}, test_error={test_error}')

    alpha = softsvmpoly(best_l, best_k, train_x, train_y)

    test_error = predict_calculate_error(alpha, train_x, test_x, test_y, best_k)
    print(f'best_k={best_k}, best_l={best_l}, test_error={test_error}')
    return best_k, best_l, test_error

def plot_error_bar_graph(x_values, averages, yerr, title, x_label, x_scale='linear', color='b', label='error bar'):
    """plots error bar graph with the given parameters"""
    # Plotting the graph with error bars
    plt.errorbar(x_values, averages,
                 yerr=yerr, fmt='-o',
                 color=color, ecolor='gray',
                 capsize=5, label=label,
                 )

    # Annotating each data point
    # for i, size in enumerate(x_values):
    #     plt.annotate(round(averages[i], ROUND_DIGITS),
    #                 # f'High: {round(errors_above[i] + averages[i], ROUND_DIGITS)}\n'
    #                 # f'Avg: {round(averages[i], ROUND_DIGITS)}\n'
    #                 # f'Low: {round(averages[i] - errors_below[i], ROUND_DIGITS)}',
    #                 (size, averages[i]),
    #                 textcoords="offset points",
    #                 xytext=(0, 10),
    #                 ha='center',
    #                 fontsize=12)

    plt.title(title, fontsize=26)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.xscale(x_scale)
    plt.xticks(x_values, fontsize=12)
    plt.grid(True)

if __name__ == '__main__':
    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    start = time.time()
    # scatter_plot('Soft SVM Samples scatter', 'X', 'Y')
    analyze_lambda_k_values(5, [1, 10, 100], [2, 5, 8], trainX, trainy, testX, testy)
    end = time.time()
    plt.show()
    print(f'time_to_process in seconds {end - start}')
