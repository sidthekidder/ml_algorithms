# implements adaboost and bagging classifier using decision tree as base estimator

import numpy as np
import os
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def entropy(y, wts=None):
    counter = {}
    for idx, i in enumerate(y):
        if i in counter:
            counter[i] += wts[idx]*1
        else:
            counter[i] = wts[idx]*1
    
    entr = 0
    for k,v in counter.items():
        entr += -(v/sum(wts)) * math.log(v/sum(wts), 2)
    
    return entr

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=[]):
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
    # initialize default weights
    if len(weights) == 0:
        weights = np.ones(len(x)) / len(x)

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

    # if all elements of list are the same, a set formed from the list will be of length 1
    if len(set(y)) <= 1:
        return y[0]
    
    # if max depth reached or no further values to split on, return majority element
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        # store a weighted counter for all unique elements
        counter = {}
        for idx, label in enumerate(y):
            if label not in counter:
                counter[label] = weights[idx]*1
            else:
                counter[label] += weights[idx]*1

        # save the label with max weight
        maj_ele = 0
        max_val = 0
        for k,v in counter.items():
            if v > max_val:
                maj_ele, max_val = k, v

        return maj_ele

    max_attr = None
    max_info_gain = 0
    cur_entropy = entropy(y, weights)

    # for each possible column/value pair, split that column into 1s and 0s based on if it is equal to the value
    # save attribute which gives max possible information gain
    for attr in attribute_value_pairs:
        column_index = attr[0]
        value_to_split_on = attr[1]
        new_column = [int(val == value_to_split_on) for val in x[:, column_index]]

        # calculate mutual information if we choose this column to split on with this value
        new_label_split_true = []
        new_label_split_true_weights = []
        new_label_split_false = []
        new_label_split_false_weights = []

        before_entropy = entropy(y, weights)
        for idx, row in enumerate(new_column):
            if row == 1:
                new_label_split_true.append(y[idx])
                new_label_split_true_weights.append(weights[idx])
            else:
                new_label_split_false.append(y[idx])
                new_label_split_false_weights.append(weights[idx])

        possible_entropy = (sum(new_label_split_true_weights)/sum(weights)) * entropy(new_label_split_true, new_label_split_true_weights) + \
                            (sum(new_label_split_false_weights)/sum(weights)) * entropy(new_label_split_false, new_label_split_false_weights)

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

def predict_item(x, tree):
    # check if leaf label reached
    if type(tree) is not dict:
        return tree

    for key in tree.keys():
        true_option = tree[(key[0], key[1], True)]
        false_option = tree[(key[0], key[1], False)]
        if x[key[0]] == key[1]:
            return predict_item(x, true_option)
        else:
            return predict_item(x, false_option)

def print_tree(tree, depth=0):
    if type(tree) is not dict:
        print(depth*"\t" + str(tree))
        return

    for idx, key in enumerate(tree):
        print(depth*"\t" + "data[" + str(key[0]) + "] == " + str(key[1]) + "? " + str(key[2]))
        print_tree(tree[key], depth+1)

def bagging(x, y, max_depth, num_trees):
    trees_ensemble = []
    for i in range(num_trees):
        # randomly sample with replacement
        sample_indexes = np.random.choice(np.arange(len(x)), len(x), replace=True)
        xsample = x[sample_indexes]
        ysample = y[sample_indexes]
        dt = id3(xsample, ysample, max_depth=max_depth)
        trees_ensemble.append(dt)

    return trees_ensemble

def adaboost(Xtrn, ytrn, max_depth, num_stumps):
    ensemble = []

    # init weights to 1/N each
    weights = np.ones(len(Xtrn)) / len(Xtrn)

    for i in range(num_stumps):
        dtree = id3(Xtrn, ytrn, max_depth=max_depth, weights=weights)

        # predict using the newly learnt stump
        y_pred = [predict_item(X, dtree) for X in Xtrn]

        # calculate error
        err = 0
        for idx, predicted_item in enumerate(y_pred):
            if predicted_item != ytrn[idx]:
                err += weights[idx]
        err /= sum(weights)

        # calculate alpha
        alpha = 0.5 * np.log((1 - err) / err)

        # save the hypothesis stump along with alpha weight
        ensemble.append((dtree, alpha))

        # recalculate weights
        new_weights = []
        for idx, weight in enumerate(weights):
            if y_pred[idx] == ytrn[idx]:
                new_weights.append(weight * np.exp(-alpha))
            else:
                new_weights.append(weight * np.exp(alpha))

        # normalize weights
        newsum = weights / (2 * np.sqrt((1 - err) * err))
        new_weights = new_weights / sum(newsum)
        weights = new_weights

    return ensemble

def predict_example(x, h_ens):
    predictions = []

    # for each testing example
    for item in x:
        # keep count of the weighted number of times each label is predicted
        options = {}
        for tree in h_ens:
            pred_label = predict_item(item, tree[0])
            if pred_label not in options:
                options[pred_label] = 0
            options[pred_label] += 1*tree[1] # multiply by weight of this ensemble

        # save the label with max weight
        selected_label = 0
        max_val = 0
        for k,v in options.items():
            if v > max_val:
                selected_label, max_val = k, v

        predictions.append(selected_label)

    return predictions


if __name__ == "__main__":
    # load training data
    dataset1 = np.genfromtxt('./mushroom.train', missing_values=0, delimiter=',', dtype=int)
    ytrn = dataset1[:, 0] # select prediction column
    Xtrn = dataset1[:, 1:] # select all other columns

    dataset2 = np.genfromtxt('./mushroom.test', missing_values=0, delimiter=',', dtype=int)
    ytst = dataset2[:, 0] # select prediction column
    Xtst = dataset2[:, 1:] # select all other columns

    # BAGGING
    print("BAGGING:")
    for depth in [3, 5]:
        for tree in [5, 10]:
            print("\nLearning ensemble for depth = " + str(depth) + " and k = " + str(tree) + "")
            ensemble = bagging(Xtrn, ytrn, max_depth=depth, num_trees=tree)
            y_pred = predict_example(Xtst, [(e, 1) for e in ensemble])

            # compute testing error
            tst_err = sum(ytst != y_pred) / len(ytst)

            print("Accuracy: " + str(100 - tst_err*100) + "%")
            print("Confusion matrix: ")
            print(confusion_matrix(ytst, y_pred))

    # BOOSTING
    print("\nBOOSTING:")
    for depth in [1, 2]:
        for stump in [5, 10]:
            print("\nLearning ensemble for depth = " + str(depth) + " and k = " + str(stump) + "...")
            ensemble = adaboost(Xtrn, ytrn, max_depth=depth, num_stumps=stump)
            y_pred = predict_example(Xtst, ensemble)

            # compute testing error
            tst_err = sum(ytst != y_pred) / len(ytst)

            print("Accuracy: " + str(100 - tst_err*100) + "%")
            print(confusion_matrix(ytst, y_pred))

    # BAGGING using scikit-learn
    print("\nBAGGING using scikit-learn:")
    for depth in [3, 5]:
        for tree in [5, 10]:
            print("\nLearning ensemble for depth = " + str(depth) + " and k = " + str(tree) + "")
            y_pred = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=depth), n_estimators=tree).fit(Xtrn, ytrn).predict(Xtst)
            #, max_depth=depth, num_trees=tree)
            # y_pred = predict_example(Xtst, [(e, 1) for e in ensemble])

            # compute testing error
            tst_err = sum(ytst != y_pred) / len(ytst)
            print("Accuracy: " + str(100 - tst_err*100) + "%")
            print("Confusion matrix: ")
            print(confusion_matrix(ytst, y_pred))

    # BOOSTING using scikit-learn
    print("\nBOOSTING using scikit-learn:")
    for depth in [1, 2]:
        for tree in [5, 10]:
            print("\nLearning ensemble for depth = " + str(depth) + " and k = " + str(tree) + "")
            y_pred = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=depth), n_estimators=tree).fit(Xtrn, ytrn).predict(Xtst)
            #, max_depth=depth, num_trees=tree)
            # y_pred = predict_example(Xtst, [(e, 1) for e in ensemble])

            # compute testing error
            tst_err = sum(ytst != y_pred) / len(ytst)

            print("Accuracy: " + str(100 - tst_err*100) + "%")
            print("Confusion matrix: ")
            print(confusion_matrix(ytst, y_pred))
    