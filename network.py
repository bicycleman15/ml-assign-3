#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
from mnist_loader import vectorized_result

class Network:
    def __init__(self, sizes, activation='sigmoid'):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Assign weights and biases randomly
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_dash = sigmoid_prime
        elif activation == 'relu':
            self.activation = Relu
            self.activation_dash = Relu_prime
        else:
            assert False, "Activation function not valid"

        self.activation_last = sigmoid
        self.activation_dash_last = sigmoid_prime

    def forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i == len(self.weights)-1:
                a = self.activation_last(np.dot(w, a) + b)
            else:
                a = self.activation(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, lr_update=None, early_stopping=False, max_iter_no_change=5):

        print("Starting training of network. architecure is",self.sizes)
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        prev_loss = 1e15
        prev_cnt = 0
        for epoch in range(epochs):

            # Shuffle training data first for SGD
            random.shuffle(training_data)

            # Create mini-batches
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            if lr_update:
                eta = lr_update(epoch)
            
            # Gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Run validation if possible
            if test_data:
                print("Epoch {} : test acc : {:.4f} | train acc : {:.4f} | lr : {:.5f} | ".format(epoch, self.evaluate(test_data, convert=False), self.evaluate(training_data, convert=True), eta), end='')
            else:
                print("Epoch {} complete.".format(epoch), end=' | ')

            cur_loss = self.total_cost(training_data)
            print("loss : {:.5f}".format(cur_loss))
            if np.sum(abs(cur_loss - prev_loss)) <= 1e-5 and early_stopping:
                prev_cnt += 1
                if prev_cnt == mx_iter_no_change:
                    print("Loss did not decrease much. Early Stopping.")
                    break
                else:
                    prev_cnt = 0
                    prev_loss = cur_loss
            else:
                prev_cnt = 0
                prev_loss = cur_loss
            
    def update_mini_batch(self, mini_batch, eta):
        sum_derivative_b = [np.zeros(b.shape) for b in self.biases]
        sum_derivative_w = [np.zeros(w.shape) for w in self.weights]

        # Add all derivatives together for each batch
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            sum_derivative_b = [Sdb + db for Sdb, db in zip(sum_derivative_b, delta_b)]
            sum_derivative_w = [Sdw + dw for Sdw, dw in zip(sum_derivative_w, delta_w)]

        # Apply the update rule
        self.weights = [w - (eta/len(mini_batch))*dw for w, dw in zip(self.weights, sum_derivative_w)]
        self.biases = [b - (eta/len(mini_batch))*db for b, db in zip(self.biases, sum_derivative_b)]

    def backprop(self, x, y):
        all_derivatives_b = [np.zeros(b.shape) for b in self.biases]
        all_derivatives_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            zs.append(z)
            if i == len(self.weights)-1:
                activation = self.activation_last(z)
            else:
                activation = self.activation(z)
            activations.append(activation)

        # NOTE : delta_z always refers to the current layer z'

        # Case 1 : Output layer
        delta_z = self.cost_derivative(activations[-1], y) * self.activation_dash_last(zs[-1])
        all_derivatives_b[-1] = delta_z
        all_derivatives_w[-1] = np.dot(delta_z, activations[-2].T)
        
        # Case 2 : All layers except last layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            a_dash = self.activation_dash(z)
            delta_z = np.dot(self.weights[-l+1].T, delta_z) * a_dash
            all_derivatives_b[-l] = delta_z
            all_derivatives_w[-l] = np.dot(delta_z, activations[-l-1].T)
        
        return (all_derivatives_b, all_derivatives_w)

    def evaluate(self, test_data, convert):
        if convert == False:
            test_results = [(np.argmax(self.forward(x)), y)
                            for (x, y) in test_data]
        else:
            test_results = [(np.argmax(self.forward(x)), np.argmax(y))
                            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def run_predictions(self, test_data):
        test_results = [np.argmax(self.forward(x)) for (x, y) in test_data]
        return test_results

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def total_cost(self, data, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            if convert: y = vectorized_result(y)
            cost += np.sum((a-y)**2)/len(data)
            # cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return 0.5*cost

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def Relu(z):
    """The ReLu function."""
    return np.maximum(z, 0)

def Relu_prime(z):
    """Derivative of the ReLu function."""
    return np.greater(z, 0).astype(np.float)

def lr_scheduler(epoch):
    n_not = 5
    return n_not / np.sqrt(epoch + 1) 