from network import Network, lr_scheduler
import mnist_loader
training_data, testing_data = mnist_loader.kannada_loader()

mid_layers = [1]
architecture = [784, 100, 100, 10]
num_epochs = 125
mini_batch_size = 100
lr = 0.5

from time import time

for mid in mid_layers:
    # architecture[1] = mid
    model = Network(sizes = architecture, activation='sigmoid')

    start = time()
    model.SGD(training_data, num_epochs, mini_batch_size, lr, test_data=testing_data, lr_update=None)
    duration = time() - start
    
    train_acc = model.evaluate(training_data, convert=True)
    test_acc = model.evaluate(testing_data, convert=False)
    
    # f = open("nn_result.txt","a")
    # print("mid={} | train acc: {:.5f} | test acc: {:.5f} | time elapsed : {:.5f}".format(mid, train_acc, test_acc, duration), file=f)
    # f.close()
    print("mid={} | train acc: {:.5f} | test acc: {:.5f} | time elapsed : {:.5f}".format(mid, train_acc, test_acc, duration))
