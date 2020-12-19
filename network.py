#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network:
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Assign weights and biases randomly
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.activation = sigmoid
        self.activation_dash = sigmoid_prime

    def forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, lr_update=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for epoch in range(epochs):

            # Shuffle training data first for SGD
            random.shuffle(training_data)

            # Create mini-batches
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Run validation if possible
            if test_data:
                print("Epoch {} : {} / {}".format(epoch,self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(epoch))
            
            if lr_update:
                eta = lr_update(eta)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        sum_derivative_b = [np.zeros(b.shape) for b in self.biases]
        sum_derivative_w = [np.zeros(w.shape) for w in self.weights]

        # Add all derivatives together for each batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            sum_derivative_b = [Sdb + db for Sdb, db in zip(sum_derivative_b, delta_nabla_b)]
            sum_derivative_w = [Sdw + dw for Sdw, dw in zip(sum_derivative_w, delta_nabla_w)]

        # Apply the update rule
        self.weights = [w - (eta/len(mini_batch))*dw for w, dw in zip(self.weights, sum_derivative_w)]
        self.biases = [b - (eta/len(mini_batch))*db for b, db in zip(self.biases, sum_derivative_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        all_derivatives_b = [np.zeros(b.shape) for b in self.biases]
        all_derivatives_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)

        # NOTE : delta_z always refers to the current layer z'

        # Case 1 : Output layer
        delta_z = self.cost_derivative(activations[-1], y) * self.activation_dash(zs[-1])
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

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))