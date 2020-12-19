from network import Network
import mnist_loader
training_data, testing_data = mnist_loader.kannada_loader()

architecture = [784, 30, 10]
num_epochs = 1000
mini_batch_size = 100
lr = 0.5

model = Network(sizes = architecture)
model.SGD(training_data, num_epochs, mini_batch_size, lr, test_data=testing_data)
