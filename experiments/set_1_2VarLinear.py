# Single linear split across two variables

# 50% of data for training, 20% for validation, 30% for testing
# Model types to choose: LRCT, OC1, CART

# Report: accuracy, confusion matrix, per-class and average F1, AUC

import numpy as np
from sklearn.model_selection import train_test_split
from LRCT import LRCTree
from sklearn.tree import DecisionTreeClassifier
import oc1
