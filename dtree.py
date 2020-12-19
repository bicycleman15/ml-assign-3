import numpy as np
import pandas as pd

class Node:
    def __init__(self, data_indices, parent, feature_index=-1, val_to_split=None):
        self.data_indices = data_indices
        self.parent = parent

        # if feature idx is -1, then it is a leaf node, 
        # else there must be a val_to_split provided
        self.feature_index = feature_index
        self.val_to_split = val_to_split
        if self.feature_index != -1:
            assert self.val_to_split is not None

    def assign_label(self, train_y):
        # Now calculate classifying label if we end up on this node
        y = train_y[self.data_indices]
        classes, class_counts = np.unique(y, return_counts = True)
        return classes[np.argmax(class_counts)]

def is_pure_node(data_y, data_indices):
    y = data_y[data_indices]
    return len(np.unique(y)) <= 1

def decide_best_feature(data_X, data_y, data_indices):
    pass

def split_on_best_feature(best_feature, val_to_split, data_X, data_y, data_indices):
    pass

def growTree(data_X, data_y, data_indices, parent=None, depth=0, max_depth=999):
    if is_pure_node(data_y, data_indices) or depth == max_depth:
        return Node(data_indices, parent, feature_index=-1)

    # Get the best feature to split on
    best_feature, info_gain, val_to_split = decide_best_feature(data_X, data_y, data_indices)
    lower_indices, upper_indices = split_on_best_feature(best_feature, val_to_split, data_X, data_y, data_indices)

    cur_node = Node(data_indices, parent, feature_index=best_feature, val_to_split=val_to_split)
    
    cur_node.left_child = growTree(data_X, data_y, lower_indices, parent=cur_node, depth=depth+1, max_depth=max_depth)
    cur_node.right_child = growTree(data_X, data_y, upper_indices, parent=cur_node, depth=depth+1, max_depth=max_depth)

    return cur_node

if __name__ == "__main__":

    data = pd.read_csv("decision_tree/test.csv")

    col_names = list(data.columns)
    data_array = np.array(data).astype(np.int32)

    X_train = data_array[:, :-1]
    y_train = data_array[:, -1]

    print(X_train)
    print(y_train)


    