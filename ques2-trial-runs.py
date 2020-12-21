from network import Network, lr_scheduler
from mnist_loader import kannada_loader

train_X = "kannada/X_train.npy"
train_y = "kannada/y_train.npy"
test_X = "kannada/X_test.npy"
test_y = "kannada/y_test.npy"
training_data, testing_data = kannada_loader(train_X, train_y, test_X, test_y)

mid_layers = [1]
architecture = [784, 1, 10]
num_epochs = 100
mini_batch_size = 100
lr = 0.5

from time import time
for mid in mid_layers:
    # architecture[1] = mid
    model = Network(sizes = architecture, activation='sigmoid')

    start = time()
    model.SGD(training_data, num_epochs, mini_batch_size, lr, test_data=testing_data, early_stopping=True)
    duration = time() - start
    
    train_acc = model.evaluate(training_data, convert=True)
    test_acc = model.evaluate(testing_data, convert=False)
    
    # f = open("nn_result.txt","a")
    # print("mid={} | train acc: {:.5f} | test acc: {:.5f} | time elapsed : {:.5f}".format(mid, train_acc, test_acc, duration), file=f)
    # f.close()
    print("mid={} | train acc: {:.5f} | test acc: {:.5f} | time elapsed : {:.5f}".format(mid, train_acc, test_acc, duration))
