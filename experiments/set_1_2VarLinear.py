# Single linear split across two variables

# 50% of data for training, 20% for validation, 30% for testing
# Model types to choose: LRCT, OC1, CART

# Report: accuracy, confusion matrix, per-class and average F1, AUC

import numpy as np
from sklearn.model_selection import train_test_split
from LRCT import LRCTree
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    # Generate the data, domain for x0 is [0, 14], domain for x1 is [0, 8]
    two_var_x = np.random.random((1000, 2))
    two_var_x[:, 0] = two_var_x[:, 0] * 14
    two_var_x[:, 1] = two_var_x[:, 1] * 8
    
