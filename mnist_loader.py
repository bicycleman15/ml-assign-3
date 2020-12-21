import numpy as np

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def kannada_loader():
    X_train = np.load(open("kannada/X_train.npy","rb"))
    X_train = X_train.reshape(-1, 28*28)
    X_train = X_train/255

    y_train = np.load(open("kannada/y_train.npy","rb"))
    y_train = y_train.reshape(-1, 1)

    X_test = np.load(open("kannada/X_test.npy","rb"))
    X_test = X_test.reshape(-1, 28*28)
    X_test = X_test/255

    y_test = np.load(open("kannada/y_test.npy","rb"))
    y_test = y_test.reshape(-1, 1)

    training_inputs = [np.reshape(x, (784, 1)) for x in X_train]
    training_results = [vectorized_result(int(y)) for y in y_train]

    testing_inputs = [np.reshape(x, (784, 1)) for x in X_test]
    testing_results = [int(y) for y in y_test]

    training_data = list(zip(training_inputs, training_results))
    testing_data = list(zip(testing_inputs, testing_results))

    return training_data, testing_data