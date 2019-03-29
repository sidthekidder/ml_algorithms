# implements the recursive id3 algorithm

import numpy as np
import os
import math
from sklearn.model_selection import train_test_split

def entropy(y):
    counter = {}
    for i in y:
        if i in counter:
            counter[i] += 1
        else:
            counter[i] = 1
    
    entr = 0
    for k,v in counter.items():
        entr += -(v/len(y)) * math.log(v/len(y), 2)
    
    return entr

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    creates a decision tree in dictionary format -
    {(3, 2, False):
        {(0, 1, False):
            {(4, 2, True): 1,
             (4, 2, False): 0},
         (0, 1, True):
            {(2, 1, True): 0,
             (2, 1, False): 1}},
     (3, 2, True): 1}
    """

    # initialize attribute-value pairs
    if attribute_value_pairs == None:
        # generate all combinations of (column, value)
        aggr = {}

        # initialize empty list for each index
        for idx, col in enumerate(x[0]):
            aggr[idx] = set()

        for row in x:
            for idx, col in enumerate(row):
                aggr[idx].add(col)

        attribute_value_pairs = []
        for k,v in aggr.items():
            for vi in v:
                attribute_value_pairs.append((k, vi))
        print("Number of possible splitting pairs are " + str(len(attribute_value_pairs)))

     # if all elements of list are the same, a set formed from the list will be of length 1
    if len(set(y)) <= 1:
        return y[0]
    
    # if max depth reached or no further values to split on, return majority element
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        # store a counter for all unique elements
        counter = {}
        for i in y:
            if i not in counter:
                counter[i] = 1
            else:
                counter[i] += 1

        # save the element with max counter
        maj_ele = 0
        max_val = 0
        for k,v in counter.items():
            if v > max_val:
                maj_ele, max_val = k, v

        return maj_ele

    max_attr = None
    max_info_gain = 0
    cur_entropy = entropy(y)

    # for each possible column/value pair, split that column into 1s and 0s based on if it is equal to the value
    # save attribute which gives max possible information gain
    for attr in attribute_value_pairs:
        column_index = attr[0]
        value_to_split_on = attr[1]
        new_column = [int(val == value_to_split_on) for val in x[:, column_index]]

        # calculate mutual information if we choose this column to split on with this value
        new_label_split_true = []
        new_label_split_false = []

        before_entropy = entropy(y)
        for idx, row in enumerate(new_column):
            if row == 1:
                new_label_split_true.append(y[idx])
            else:
                new_label_split_false.append(y[idx])

        possible_entropy = (len(new_label_split_true)/len(y)) * entropy(new_label_split_true) + \
                            (len(new_label_split_false)/len(y)) * entropy(new_label_split_false)

        mutual_info = abs(before_entropy - possible_entropy)

        if (mutual_info > max_info_gain):
            max_info_gain, max_attr = mutual_info, attr

    # remove the selected next max attribute-value pair from the list of pairs
    new_attribute_value_pairs = attribute_value_pairs.copy()
    new_attribute_value_pairs.remove(max_attr)

    # separate previous dataset into two datasets, based on rows which satisfy attr
    x_true_elements = []
    x_false_elements = []
    y_true_elements = []
    y_false_elements = []

    for idx, val in enumerate(x):
        if val[max_attr[0]] == max_attr[1]:
            x_true_elements.append(val)
            y_true_elements.append(y[idx])
        else:
            x_false_elements.append(val)
            y_false_elements.append(y[idx])

    x_true_elements = np.asarray(x_true_elements)
    x_false_elements = np.asarray(x_false_elements)

    # set the key as specified in comments above and value as recursive call to id3
    max_attr_true = (max_attr[0], max_attr[1], True)
    max_attr_false = (max_attr[0], max_attr[1], False)
    tree = {}
    tree[max_attr_true] = id3(x_true_elements, y_true_elements, new_attribute_value_pairs.copy(), depth+1, max_depth)
    tree[max_attr_false] = id3(x_false_elements, y_false_elements, new_attribute_value_pairs.copy(), depth+1, max_depth)

    return tree

def predict_example(x, tree):
    # check if leaf label reached
    if type(tree) is not dict:
        return tree

    for key in tree.keys():
        true_option = tree[(key[0], key[1], True)]
        false_option = tree[(key[0], key[1], False)]
        if x[key[0]] == key[1]:
            return predict_example(x, true_option)
        else:
            return predict_example(x, false_option)

def print_tree(tree, depth):
    if type(tree) is not dict:
        print(depth*"\t" + str(tree))
        return

    for idx, key in enumerate(tree):
        print(depth*"\t" + "data[" + str(key[0]) + "] == " + str(key[1]) + "? " + str(key[2]))
        print_tree(tree[key], depth+1)

# load training data
M = np.genfromtxt('./Skin_NonSkin.txt', missing_values=0, delimiter='\t', dtype=int)
ytrn = M[:, 3] # select prediction column
Xtrn = M[:, :3] # select all other columns

Xtrn, Xtst, ytrn, ytst = train_test_split(Xtrn, ytrn, test_size=0.999, random_state=42)

# learn decision tree
print("Starting learning using id3 recursive algorithm.")
decision_tree = id3(Xtrn, ytrn, max_depth=3)
print("Decision tree representation:\n")
print_tree(decision_tree, 0)

# predict examples
y_pred = [predict_example(x, decision_tree) for x in Xtst]

# compute average error
diff = 0
for idx, label in enumerate(ytst):
    if ytst[idx] != y_pred[idx]:
        diff += 1
tst_err = diff/len(ytst)

print("\nTest error: " + str(tst_err*100) + "%.")
