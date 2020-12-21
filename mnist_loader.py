import numpy as np

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def kannada_loader():
    """Return a tuple containing ``(training_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``test_data`` is list containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means I am using slightly different formats for
    the training data and the validation / test data."""

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