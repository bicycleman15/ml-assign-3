import numpy as np
import pandas as pd
from tqdm import tqdm

# This list is hardcoded
discrete_features = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, \
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, \
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53])

class Node:
    def __init__(self, train_y, parent, feature_index=-1, val_to_split=None):
        self.parent = parent
        # if feature idx is -1, then it is a leaf node, 
        # else there must be a val_to_split provided
        self.feature_index = feature_index
        self.val_to_split = val_to_split
        if self.feature_index != -1:
            assert self.val_to_split is not None

        self.left_child = None
        self.right_child = None
        
        self.assign_label(train_y)

    def assign_label(self, train_y):
        # Now calculate classifying label if we end up on this node
        if len(train_y) == 0:
            self.label = 1 # NOTE : maybe assing some random label, IDK
            return
        classes, class_counts = np.unique(train_y, return_counts = True)
        self.label = classes[np.argmax(class_counts)]

def is_pure_node(data_y):
    return len(np.unique(data_y)) <= 1

def calculateEntropy(data):
    _, uniqueClassesCounts = np.unique(data, return_counts = True)
    probabilities = uniqueClassesCounts / uniqueClassesCounts.sum()
    return sum(probabilities * -np.log2(probabilities))

def calculateOverallEntropy(lower_data, upper_data):
    plower_data = len(lower_data) / (len(lower_data) + len(upper_data))
    pupper_data = len(upper_data) / (len(lower_data) + len(upper_data))
    return plower_data * calculateEntropy(lower_data) + pupper_data * calculateEntropy(upper_data)

def find_median(train_X, feature_idx):
    sorted_X = np.sort(train_X[:, feature_idx])
    return sorted_X[len(sorted_X)//2]

def decide_best_feature(data_X, data_y):
    num_features = data_X.shape[1]
    min_entropy = float('inf')
    best_feature = None
    value_to_split = None

    for i in range(num_features):
        median = 0 if i in discrete_features else find_median(data_X, feature_idx=i)
        lower_indices, upper_indices = split_on_feature(i, median, data_X, data_y)

        lower_data = data_y[lower_indices]
        upper_data = data_y[upper_indices]
        cur_entropy = calculateOverallEntropy(lower_data, upper_data) 

        if cur_entropy < min_entropy:
            min_entropy = cur_entropy
            best_feature = i
            value_to_split = median

    assert best_feature is not None and value_to_split is not None
    return best_feature, value_to_split

def split_on_feature(feature_idx, val_to_split, data_X, data_y):
    indices_left = np.argwhere(data_X[:, feature_idx] <= val_to_split).ravel()
    indices_right = np.argwhere(data_X[:, feature_idx] > val_to_split).ravel()
    return indices_left, indices_right

def growTree(data_X, data_y, parent=None, depth=0, max_depth=999, min_data=50):
    if is_pure_node(data_y) or depth == max_depth or data_X.shape[0] <= min_data:
        return Node(data_y, parent, feature_index=-1)

    # Get the best feature to split on
    best_feature, val_to_split = decide_best_feature(data_X, data_y)
    lower_indices, upper_indices = split_on_feature(best_feature, val_to_split, data_X, data_y)

    cur_node = Node(data_y, parent, feature_index=best_feature, val_to_split=val_to_split)
    cur_node.left_child = growTree(data_X[lower_indices], data_y[lower_indices], parent=cur_node, depth=depth+1, max_depth=max_depth)
    cur_node.right_child = growTree(data_X[upper_indices], data_y[upper_indices], parent=cur_node, depth=depth+1, max_depth=max_depth)

    return cur_node

def predict(X, root_node:Node):
    """predict label on a single data point X"""
    if root_node.left_child is None or root_node.right_child is None:
        return root_node.label

    best_feature, median_val = root_node.feature_index, root_node.val_to_split

    if X[best_feature] <= median_val:
        return predict(X, root_node.left_child)
    else: 
        return predict(X, root_node.right_child)

def load_data(split='train'):
    data = pd.read_csv("decision_tree/{}.csv".format(split))
    data_array = np.array(data).astype(np.int32)
    X_data = data_array[:, :-1]
    y_data = data_array[:, -1]
    return X_data, y_data


if __name__ == "__main__":

    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')

    root_node = growTree(X_train, y_train, max_depth=15)

    correct = 0
    total = 0
    for i in tqdm(range(len(X_val))):
        if predict(X_val[i], root_node) == y_val[i]:
            correct += 1
        total += 1
    
    print(correct/total)
    