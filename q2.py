from network import Network, lr_scheduler
from mnist_loader import kannada_loader

def parse_layers(layers):
    hidden = list(map(int, layers.split()))
    final_layers = [784]
    for x in hidden:
        final_layers.append(x)
    final_layers.append(10)
    return final_layers

if __name__ == '__main__':

    train_X = "kannada/X_train.npy"
    train_y = "kannada/y_train.npy"
    test_X = "kannada/X_test.npy"
    test_y = "kannada/y_test.npy"
    output_path = "output-2.txt"

    batch_size = 100
    hidden_layers = "20 10"
    activation_type = "relu"
    num_epochs = 5
    lr = 0.5

    training_data, testing_data = kannada_loader(train_X, train_y, test_X, test_y)
    hidden_layers = parse_layers(hidden_layers)

    model = Network(sizes=hidden_layers, activation=activation_type)
    model.SGD(  training_data, 
                num_epochs, 
                batch_size, 
                lr, 
                early_stopping=True, 
                max_iter_no_change=5
            )

    predictions = model.run_predictions(testing_data)

    f = open(output_path, "w")
    for p in predictions:
        print(p, file=f)
    f.close()