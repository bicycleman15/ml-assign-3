import sys
import numpy as np
from dtree import load_data, growTree, find_accuracy, run_predictions_on_data, prune_tree_iterations

if __name__ == '__main__':

    train_path = 'decision_tree/train.csv'
    val_path = 'decision_tree/val.csv'
    test_path = 'decision_tree/test.csv'
    question = '2'
    output_path = 'output-1.txt'
    
    print(sys.argv)
    question, train_path, val_path, test_path, output_path = sys.argv[1:]

    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)

    from time import time
    start = time()
    if question == '1':
        root_node = growTree(X_train, y_train, max_depth=23)
        predictions = run_predictions_on_data(X_test, root_node)

        f = open(output_path,"w")
        for p in predictions:
            print(p, file=f)
        f.close()

    elif question == '2':
        root_node = growTree(X_train, y_train, max_depth=8)
        root_node = prune_tree_iterations(root_node, X_val, y_val)

        predictions = run_predictions_on_data(X_test, root_node)

        f = open(output_path,"w")
        for p in predictions:
            print(p, file=f)
        f.close()

    duration = time() - start
    print("time taken : {:.3f} secs.".format(duration))
