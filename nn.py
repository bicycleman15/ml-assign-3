import numpy as np
from tqdm import tqdm
from utils import softmax

def data_loader(data, batch_size):
    """Takes in data array and yields batch_size arrays
    after shuffling data
    """
    np.random.shuffle(data)
    len_data = data.shape[0]
    num_batches = len_data // batch_size
    if len_data % batch_size:
        num_batches += 1
    
    for i in range(num_batches):
        yield data[i * batch_size : (i+1) * batch_size, : ]


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def squared_loss(y_hat, y):
    assert y_hat.shape == y.shape
    loss = (y_hat - y)**2
    return np.sum(loss)

def ReLu(x):
    return np.maximum(x, 0)

def ReLu_prime(x):
    x = x.copy()
    x[x<=0] = 0.
    x[x>0] = 1.
    return x

#Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

class NueralNet:
    def __init__(self, layer_struct : list, lr, batch_size):
        """ layer struct : expects input size in first,
                            output dim in last,
                            and hidden layer sizes in intermediate elements. 
            lr : learning rate
            batch_size : used only for training
        """

        self.parameters = {}
        self.derivatives = {}

        # Make a copy of layer architecure
        self.nn = list(layer_struct)
        self.L = len(self.nn) - 1

        # Create weight matrixes and output nuerons
        for i in range(1, len(layer_struct)):
            self.parameters['W'+str(i)] = np.random.randn(self.nn[i-1], self.nn[i]) * 0.01
            self.parameters['b'+str(i)] = np.random.randn(self.nn[i]) 
        
        self.activation = sigmoid
        self.deriv_activation = sigmoid_prime
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, X):
        """ Passes X through the nueral network
        Make sure shape of X is [batch_size, no_of_input_features]

        Returns the logits
        """
        self.parameters['a0'] = X

        for i in range(1, len(self.nn)):
            self.parameters['z'+str(i)] = np.dot(self.parameters['a'+str(i-1)], self.parameters['W'+str(i)]) + self.parameters['b'+str(i)]
            self.parameters['a'+str(i)] = self.activation(self.parameters['z'+str(i)])

        # Take softmax of last layer
        self.parameters['a'+str(self.L)] = softmax(self.parameters['z'+str(self.L)], axis=1)

        # Return only the logits
        return self.parameters['z' + str(self.L)]

    def backward(self, y_gt):
        """ Applies the backpropogation on the current `aL` neurons and `y` label.
        Gradients are collected and summed. Make sure to call self.clear_grad to set grads to zero before
        calling on new batch.
        """
        assert y_gt.shape[0] == self.batch_size
        assert y_gt.shape[1] == self.nn[self.L]
        
        self.derivatives['dz' + str(self.L)] = (self.parameters['a' + str(self.L)] - y_gt)
        self.derivatives['dW' + str(self.L)] = np.dot(np.transpose(self.parameters['a' + str(self.L - 1)]), self.derivatives['dz' + str(self.L)]) # [nn[L-1], nn[L]] = [1, nn[L-1]]^T * [1, nn[L]]
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        for l in range(self.L - 1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(self.derivatives['dz' + str(l + 1)], np.transpose(self.parameters['W' + str(l + 1)])) * self.deriv_activation(self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(np.transpose(self.parameters['a' + str(l - 1)]), self.derivatives['dz' + str(l)])
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
    
    def step(self):
        """ Does the gradient descent step, by subtracting grads from params
        """
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= self.lr * np.mean(self.derivatives['dW' + str(l)], axis=0)
            self.parameters['b' + str(l)] -= self.lr * np.mean(self.derivatives['db' + str(l)], axis=0)

    def zero_grad(self):
        """Clear out the gradients"""
        self.derivatives = {}

# Test functions to check NN class
if __name__ == "__main__":
    X_train = np.load(open("kannada/X_train.npy","rb"))
    X_train = X_train.reshape(-1, 28*28)

    y_train = np.load(open("kannada/y_train.npy","rb"))
    y_train = y_train.reshape(-1, 1)

    train_data = np.concatenate([X_train, y_train], axis=1)

    X_test = np.load(open("kannada/X_test.npy","rb"))
    X_test = X_test.reshape(-1, 28*28)

    y_test = np.load(open("kannada/y_test.npy","rb"))
    y_test = y_test.reshape(-1, 1)

    test_data = np.concatenate([X_test, y_test], axis=1)

    num_epochs = 50
    batch_size = 100
    num_classes = 10
    lr = 0.005
    architecture = [784, 100, 10]

    model = NueralNet(layer_struct=architecture, lr=lr, batch_size=batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0

        correct = 0
        total = 0

        for data in data_loader(train_data, batch_size):
            model.zero_grad()

            X = data[:, :-1]
            # print(X)
            y = data[:, -1].ravel()

            # now binary vectorize y
            y_one_hot = np.eye(num_classes)[y]
            y_hat = model.forward(X)

            probs = softmax(y_hat, axis=1)
            preds = np.argmax(y_hat, axis=1).ravel()

            # print(preds.shape, y.shape)

            mask = preds == y
            correct += sum(mask)
            total += len(mask)

            model.backward(y_one_hot)
            model.step()

            epoch_loss += squared_loss(probs, y_one_hot)
            # import pdb; pdb.set_trace()
        
        # print(model.derivatives['dW1'])
        # import pdb; pdb.set_trace()
        print("Epoch {} loss : {:.5f}".format(epoch, epoch_loss / train_data.shape[0]), end=' ')
        print("| accuracy : {:.5f}".format(correct / total), end=' ')

        X = test_data[:, :-1]
        y = test_data[:, -1].ravel()

        # now binary vectorize y
        y_one_hot = np.eye(num_classes)[y]
        y_hat = model.forward(X)

        preds = np.argmax(y_hat, axis=1).ravel()
        mask = preds == y
        correct += sum(mask)
        total += len(mask)

        print("| val accuracy : {:.5f}".format(correct / total))