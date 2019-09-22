#Decision tree Coursework

from io import StringIO
import numpy as np
import random
import math
import copy

#clean_data = np.loadtxt("clean_dataset.txt")
#noisy_data = np.loadtxt("noisy_dataset.txt")

# print(clean_data)
# print(noisy_data)
#
# print(clean_data.shape)
# print(noisy_data.shape)

LABEL_INDEX = 7 # index of label is 7

flatten = lambda l: [item for sublist in l for item in sublist]

def probability(label, dataset):
    number_of_label = 0
    for row in dataset:
        if row[LABEL_INDEX] == label:
            number_of_label += 1
    return number_of_label/len(dataset)

def H(dataset):
    labels = set([row[LABEL_INDEX] for row in dataset])
    result = 0
    for label in labels:
        result += probability(label, dataset)*math.log2(probability(label, dataset))
    return result*-1

def Halfremainder(dataset1, dataset2):
    return len(dataset1)/(len(dataset2)+len(dataset1))*H(dataset1)

def Remainder(Sleft, Sright):
    return Halfremainder(Sleft, Sright) + Halfremainder(Sright, Sleft)

def Gain(Sall, Sleft, Sright):
    return H(Sall) - Remainder(Sleft, Sright)

# get all values appear in the column
def unique_values(column, dataset):
    return set(row[column] for row in dataset)

# return true of all samples in the dataset have same label
def same_label(dataset):
    label = dataset[0][LABEL_INDEX] # take the label from the first sample
    for sample in dataset:
        if sample[LABEL_INDEX] != label:
            return False
    return True

# return the depth of the tree
def find_depth(tree):
    if isinstance(tree, dict):
        return 1 + (max(map(find_depth, tree.values())) if tree else 0)
    return -1

# split dataset into left(True) and right(Flase)
def split_data(dataset, value, column):
    Sleft = []
    Sright = []

    for row in dataset:
        if (row[column] <= value):
            Sleft.append(row)
        else:
            Sright.append(row)

    return (Sleft, Sright)

# return (attribute index, value) for the given dataset
def find_split(training_dataset):
    best_attr = 0
    best_gain = 0
    best_value = 0

    # test all values for each attribute
    for i in range(7):
        # TODO: do we need to try ALL values? (ie. is there more efficient way?)
        for value in unique_values(i, training_dataset):
            # split dataset into left(True) and right(Flase)
            (Sleft, Sright) = split_data(training_dataset, value, i)

            gain = Gain(training_dataset, Sleft, Sright)

            # update best_gain (for this attribute)
            if gain > best_gain:
                best_gain = gain
                best_value = value
                best_attr = i

    return (best_attr, best_value)

def decision_tree_learning(training_dataset, depth):
    if same_label(training_dataset):
        leaf_node = {'label': training_dataset[0][LABEL_INDEX], 'leaf': True}
        return (leaf_node, depth)
    else:
        (attr, value) = find_split(training_dataset)
        (Sleft, Sright) = split_data(training_dataset, value, attr)

        (l_branch, l_depth) = decision_tree_learning(Sleft, depth+1)
        (r_branch, r_depth) = decision_tree_learning(Sright, depth+1)

        node = {'attribute': attr, 'value': value, 'left': l_branch, 'right': r_branch, 'leaf': False, 'checked': False}

        return (node, max(l_depth, r_depth))

# split the given dataset into 10 folds
def get_folds(dataset):
    np.random.shuffle(dataset) # shuffle the data
    folds = []
    fold_size = int(len(dataset)/10)
    for i in range(10):
        folds.append(dataset[i*fold_size : (1+i)*fold_size].tolist())
    return folds

# loop with k folds of (validation set & training set) to choose the best tree
def parameter_tuning(folds, k):
    best_accuracy = 0
    best_tree = {}
    best_depth = 0

    for i in range(k):
        validation_dataset = folds[i]
        training_dataset = flatten(folds[0:i] + folds[i+1:])
        (tree, depth) = decision_tree_learning(training_dataset, 0)
        accuracy = evaluate(validation_dataset, tree)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_tree = tree
            best_depth = depth

    return (best_tree, best_depth)

# implement 10-fold cross validation on the given dataset
def cross_validation(dataset):
    folds = get_folds(dataset)
    accuracy_list = []
    recall_list = np.zeros((4, 10))
    precision_list = np.zeros((4, 10))
    trees = []
    depths = []

    for i in range(10):
        test_dataset = folds[i]
        (tree, depth) = parameter_tuning(folds[0:i] + folds[i+1:], 9)
        for j in range(4):
            recall_value = recall(j, test_dataset, tree)
            recall_list[j][i] = recall_value
            precision_value = precision(j, test_dataset, tree)
            precision_list[j][i] = precision_value
        accuracy = evaluate(test_dataset, tree)
        accuracy_list.append(accuracy)
        trees.append(tree)
        depths.append(depth)

    average_accuracy = float(sum(accuracy_list))/10#
    average_recall = []
    average_precision = []
    average_f = []

    for k in range(4):
        average_recall.append(np.sum(recall_list[k])/10)
        average_precision.append(np.sum(precision_list[k])/10)
        average_f.append(F_measure(average_recall[k], average_precision[k]))
    return average_accuracy, average_recall, average_precision, average_f, trees, depths


# test the sample(row) on the tree, return estimated label of the given sample
def test_sample(sample, node):
    if node['leaf']:
        return node['label']
    elif sample[node['attribute']] < node['value']:
        return test_sample(sample, node['left'])
    else:
        return test_sample(sample, node['right'])


# compute (TP, FN, FP, TN) for the given label on the testset
def classification_result(test_dataset, trained_tree):
    confusion_matrix = np.zeros((4, 4))
    labels = set([row[LABEL_INDEX] for row in test_dataset])
    for row in test_dataset:
        predictedlabel = test_sample(row, trained_tree)
        a, b = int(predictedlabel)-1, int(row[LABEL_INDEX])-1
        confusion_matrix[a][b] +=1
    return confusion_matrix

# return the accuracy of the trained tree on test dataset
def evaluate(test_dataset, trained_tree):
    confusion_matrix = classification_result(test_dataset, trained_tree)
    return np.trace(confusion_matrix)/len(test_dataset)

# return average recall for each label
def recall(label, test_dataset, trained_tree):
    row_sum = 0
    confusion_matrix = classification_result(test_dataset, trained_tree)
    if(np.sum(confusion_matrix[label-1]) != 0):
        row_sum += confusion_matrix[label-1][label-1] / np.sum(confusion_matrix[label-1])
    return row_sum

def precision(label, test_dataset, trained_tree):
    col_sum = 0
    confusion_matrix = classification_result(test_dataset, trained_tree)
    if(np.sum(confusion_matrix[label-1][label-1]) != 0):
        col_sum += confusion_matrix[label-1][label-1] / np.sum(confusion_matrix[:, label-1])
    return col_sum

def F_measure(precision_value, recall_value, alpha = 1):
    return (1 + alpha*alpha)*(precision_value*recall_value)/((alpha*alpha*precision_value)+recall_value)

# return a boolean whether the node diretly connects two leaves
def node_with_2leaves(node):
    if node['leaf']:
        return False
    else:
        return (node['left']['leaf'] and node['right']['leaf'])

# return the most popular class at the node with two leaves
def majority_class(samples, node):
    (Sleft, Sright) = split_data(samples, node['value'], node['attribute'])
    if len(Sleft) >=  len(Sright):
        return node['left']['label']
    else:
        return node['right']['label']

# no need to prune on this tree if each branch is marked/is a leaf (but not both leaves)
def nodes_checked(tree):
    if node_with_2leaves(tree):
        return false
    elif tree['left']['leaf']:
        return tree['right']['checked']
    elif tree['right']['leaf']:
        return tree['left']['checked']
    else:
        return tree['left']['checked'] and tree['right']['checked']

# prune one node from the (sub)tree. The input tree should NEVER be a "leaf"
# marked_tree is used for marking the node in the case pruned_tree does not reduce error
# check through the tree from left branch
def prune_node(samples, tree):
    marked_tree = copy.deepcopy(tree)
    pruned_tree = copy.deepcopy(tree)

    (Sleft, Sright) = split_data(samples, tree['value'], tree['attribute'])

    if nodes_checked(tree):
        marked_tree['checked'] = True
        return (marked_tree, marked_tree)

    if not tree['left']['leaf'] and not tree['left']['checked']:
        if node_with_2leaves(tree['left']):
            label = majority_class(Sleft, tree['left'])
            pruned_tree['left'] = {'label': label, 'leaf': True}
            marked_tree['left']['checked'] = True
            return (pruned_tree, marked_tree)
        else:
            (pruned_left_tree, marked_left_tree) = prune_node(Sleft, tree['left'])
            pruned_tree['left'] = pruned_left_tree
            marked_tree['left'] = marked_left_tree
            return (pruned_tree, marked_tree)

    if not tree['right']['leaf'] and not tree['right']['checked']:
        if node_with_2leaves(tree['right']):
            label = majority_class(Sright, tree['right'])
            pruned_tree['right'] = {'label': label, 'leaf': True}
            marked_tree['right']['checked'] = True
            return (pruned_tree, marked_tree)
        else:
            (pruned_right_tree, marked_right_tree) = prune_node(Sright, tree['right'])
            pruned_tree['right'] = pruned_right_tree
            marked_tree['right'] = marked_right_tree
            return (pruned_tree, marked_tree)
    else:
        # no need to prune on this tree
        marked_tree['checked'] = True
        return(marked_tree, marked_tree)

# test if the pruned tree reduces error
def is_prunedtree_better(original_tree, pruned_tree, test_dataset):
    return evaluate(test_dataset, pruned_tree) >= evaluate(test_dataset, original_tree)

# prune the tree according to the test dataset
def pruning(test_dataset, tree):
    if node_with_2leaves(tree):
        # the tree has only one node
        label = majority_class(test_dataset, tree)
        pruned_tree = {'label': label, 'leaf': True}
        if is_prunedtree_better(tree, pruned_tree, test_dataset):
            return pruned_tree
        else:
            tree['checked'] = True
            return tree

    elif nodes_checked(tree):
        # no node need to be pruned
        return tree
    else:
        (pruned_tree, marked_tree) = prune_node(test_dataset, tree)
        # (pruned_tree == marked_tree) indicates no need to prune
        if pruned_tree == marked_tree or \
            is_prunedtree_better(tree, pruned_tree, test_dataset):
            # pruning the node reduces error rate
            return pruning(test_dataset, pruned_tree)
        else:
            # pruning the node does NOT reduce error rate
            return pruning(test_dataset, marked_tree)

# loop with k folds of (validation set & testing set) to find best pruning on the given tree
# use two of the k folds as validation set and testing set respectively
def parameter_tuning_pruning(dataset, k, tree):
    folds = get_folds(dataset)
    best_accuracy = 0
    best_pruned_tree = {}

    for i in range(k):
        test_dataset = folds[i]
        validation_dataset = folds[i-1]
        pruned_tree = pruning(validation_dataset, tree)
        accuracy = evaluate(test_dataset, pruned_tree)
        if accuracy > best_accuracy:
            best_pruned_tree = pruned_tree
            best_accuracy =  accuracy

    return best_pruned_tree

def cross_validation_tuning(dataset, trees):
    folds = get_folds(dataset)
    accuracy_list = []
    recall_list = np.zeros((4, 10))
    precision_list = np.zeros((4, 10))

    for i in range(10):
        test_dataset = folds[i]
        tree = trees[i];
        for j in range(4):
            recall_value = recall(j, test_dataset, tree)
            recall_list[j][i] = recall_value
            precision_value = precision(j, test_dataset, tree)
            precision_list[j][i] = precision_value
        accuracy = evaluate(test_dataset, tree)
        accuracy_list.append(accuracy)

    average_accuracy = float(sum(accuracy_list))/10#
    average_recall = []
    average_precision = []
    average_f = []

    for k in range(4):
        average_recall.append(np.sum(recall_list[k])/10)
        average_precision.append(np.sum(precision_list[k])/10)
        average_f.append(F_measure(average_recall[k], average_precision[k]))
    return average_accuracy, average_recall, average_precision, average_f



# printing part : creating a tree and printing informations out of it.

def Tree_info_printing(dataset):

    avg_acc, avg_rec, avg_p, avg_f, trees, depths = cross_validation(dataset)
    avg_accp, avg_recp, avg_pp, avg_fp =cross_validation_tuning(dataset, trees)
    print("depth",  depths)
    print("find depth", find_depth(trees[0]))
    print("average accuracy", avg_acc)
    print("average recall", avg_rec)
    print("average precision", avg_p)
    print("average F-1", avg_f)
    print("find depth", find_depth(trees[0]))
    print("average  pruned accuracy", avg_accp)
    print("average pruned recall", avg_recp)
    print("average pruned precision", avg_pp)
    print("average pruned F-1", avg_fp)

# Info printing call
#print("clean_data")
#Tree_info_printing(clean_data)
#print("noisy_data")
#Tree_info_printing(noisy_data)

# def print_tree(tree):
    # print out the decision tree
