import mnist_loader
from sklearn.neural_network import MLPClassifier
import numpy as np


training_data, testing_data = mnist_loader.kannada_loader()

model = MLPClassifier(hidden_layer_sizes=[784, 100, 100, 10], 
                        activation='relu', 
                        solver='sgd', 
                        verbose=True,
                        batch_size=100,
                        learning_rate='adaptive',
                        max_iter=100,
                        early_stopping=True
                    )

# First transform training and testing data
train_inputs = np.array([x[0].ravel() for x in training_data])
train_gts = np.array([np.argmax(x[1].ravel()) for x in training_data]).astype(np.int)

test_inputs = np.array([x[0].ravel() for x in testing_data])
test_gts = np.array([x[1] for x in testing_data]).astype(np.int)

model.fit(train_inputs, train_gts)
print("Training metrics:")
print(model.score(train_inputs, train_gts))
print("Testing metrics:")
print(model.score(test_inputs, test_gts))
