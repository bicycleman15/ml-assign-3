import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def squared_loss(y_hat, y):
    assert y_hat.shape == y.shape
    loss = (y_hat - y)**2
    return np.sum(loss)

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
            self.parameters['W'+str(i)] = np.zeros([self.nn[i-1], self.nn[i]])
            self.parameters['b'+str(i)] = np.zeros(self.nn[i]) 
        
        self.activation = sigmoid
        self.deriv_activation = sigmoid_prime
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, X):
        """ Passes X through the nueral network
        Make sure shape of X is [batch_size, no_of_input_features]
        """
        self.parameters['a0'] = X

        for i in range(1, len(self.nn)):
            self.parameters['z'+str(i)] = np.dot(self.parameters['a'+str(i-1)], self.parameters['W'+str(i)]) + self.parameters['b'+str(i)]
            self.parameters['a'+str(i)] = self.activation(self.parameters['z'+str(i)])

        return self.parameters['a' + str(len(self.nn)-1)]

    def backward(self, y):
        """ Applies the backpropogation on the current `aL` neurons and `y` label.
        Gradients are collected and summed. Make sure to call self.clear_grad to set grads to zero before
        calling on new batch.
        """
        assert y.shape[0] == self.batch_size
        assert y.shape[1] == self.nn[self.L]
        
        self.derivatives['dz' + str(self.L)] = (self.parameters['a' + str(self.L)] - y) * self.deriv_activation(self.parameters['z' + str(self.L)])
        self.derivatives['dW' + str(self.L)] = np.dot(np.transpose(self.parameters['a' + str(self.L - 1)]), self.derivatives['dz' + str(self.L)]) # [nn[L-1], nn[L]] = [1, nn[L-1]]^T * [1, nn[L]]
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        for l in range(self.L - 1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(self.derivatives['dz' + str(l + 1)], np.transpose(self.parameters['W' + str(l + 1)])) * self.deriv_activation(self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(np.transpose(self.parameters['a' + str(l)]), self.derivatives['dz' + str(l)])
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
    
    def step(self):
        """ Does the gradient descent step, by subtracting grads from params
        """
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= self.lr * (np.sum(self.derivatives['dW' + str(l)], axis=0) / self.batch_size)
            self.parameters['b' + str(l)] -= self.lr * (np.sum(self.derivatives['db' + str(l)], axis=0) / self.batch_size)

    def zero_grad(self):
        """Clear out the gradients for a new batch"""
        self.derivatives = {}

# Test functions to check NN class
if __name__ == "__main__":
    model = NueralNet(layer_struct=[2, 3, 1], batch_size=4, lr=1)

    data = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,1]])

    import random
    for i in range(10000):
        model.zero_grad()
        loss = 0
       
        X = data[:, :2]
        y = np.array(data[:, 2])
        y = y.reshape(-1,1)
        y_hat = model.forward(X)

        model.backward(y)
        loss += squared_loss(y_hat, y)
        
        model.step()
        print("Epoch loss : {:.5f}".format(loss))
    # y = model.forward(x)  

    y = model.forward(np.array([[0, 0]]))
    print(y)

    y = model.forward(np.array([[0, 1]]))
    print(y)

    y = model.forward(np.array([[1, 0]]))
    print(y)

    y = model.forward(np.array([[1, 1]]))
    print(y)
