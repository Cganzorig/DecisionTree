////////////////////////////////////README//////////////////////////////////

Our code has been realized in a linux environment (lab machines) using Python3.

Since approximately 90 trees are created while running the cross validation,

running might take some time. To cancel, use <Ctrl + c> in the terminal used to

run the executable.

///////////////////////////////////////

For our code to work, a dataset loaded with np.loadtxt is needed. Since the
datasets are not included in the .tar, uncomment and update the following lines:

#clean_data = np.loadtxt("clean_dataset.txt")
#noisy_data = np.loadtxt("noisy_dataset.txt")

actualising the pathes is necessary if the files are not copied in the same
directory than our source code.

the concerned lines are the first lines of code after the import Session.

///////////////////////////////////////

To run the code, please use the command: (Execution without print)

python coursework.py3

This will return the following information for both unpruned trees and pruned trees:
List of depths of 10 trees created during cross validation
Average accuracy
Average recall
Average precision
Average F-1

///////////////////////////////////////

For testing, please replace the loading path:

noisy_data = np.loadtxt("%noisy_dataset.txt%")

changing %...% by your dataset name/path.

You could also replace the parameter in the last line of our code:

Tree_info_printing(noisy_data)

by a loaded dataset using np.loadtxt.

///////////////////////////////////////

To obtain a similar output of the results displayed in the appendix session,
please uncomment the  4 lines under # Info printing call.

///////////////////////////////////////

Our code is segmented according to the following:

# SECTION 1 : regroups the calculations needed to the Gain definition
(see specifications)

# SECTION 2 : set of functions used as helper for the decision_tree_learning
function, including the decision_tree_learning itself.

# SECTION 3 : Set of functions used as helper for cross_validation :
functions before cross validation are used to generate the tree,
functions after it are used to calculate Recall, accuracy, precision...

# SECTION 4 : Set of functions used as helper for the pruning function.

# SECTION 5 : cross validation for prunned tree

# SECTION 6 : printing helper function, takes only a dataset as parameter.
