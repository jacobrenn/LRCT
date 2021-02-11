# Checkerboard configuration experiment, 2 variables

# 50% of data for training, 20% for validation, 30% for testing
# Model types to choose: LRCT, OC1, CART, KNN, LogisticRegression, Fully-Connected Neural Network

# Report: accuracy, confusion_matrix, per-class and average F1, AUC

import numpy as np
from sklearn.model_selection import train_test_split
from LRCT import LRCTree
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import PredefinedSplit, GridSearchCV

if __name__ == '__main__':
    # Generate the ata, domain for x0 and x1 is [0, 10]
    np.random.seed(4736)
    two_var_x = np.random.random((10000, 2))
    two_var_x = two_var_x * 10
    target = (np.floor(two_var_x[:, 0]) + np.floor(two_var_x[:, 1])) % 2

    # Create train, val, test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(two_var_x, target, test_size = 0.3, random_state = 133523)
    val_indices = np.random.choice(x_train_val.shape[0], int(x_train_val.shape[0] * 0.4), replace = False)
    train_val_fold = [0 if i in val_indices else -1 for i in range(x_train_val.shape[0])]
    ps = PredefinedSplit(test_fold = train_val_fold)
    
    # Plot the data
    plt.figure(figsize = (10, 4))
    plt.scatter(
        x_train_val[train_val_fold == -1][:, 0],
        x_train_val[train_val_fold == -1][:, 1],
        c = y_train_val[train_val_fold == -1],
        cmap = 'Set1'
    )
    plt.title('Training Data, Experiment 9', fontsize = 'xx-large')
    plt.xlabel('Feature 1', fontsize = 'x-large')
    plt.ylabel('Feature 2', fontsize = 'x-large')
    plt.xticks(fontsize = 'large')
    plt.yticks(fontsize = 'large')
    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.5, 10.5)
    plt.savefig('exp9/exp_9_training.png')

    
    lrct_params = {
        'max_depth': range(8, 12),
        'min_samples_split': range(5, 1, -1),
        'min_samples_leaf': range(5, 1, -1),
        'highest_degree': range(1, 3),
        'n_bins': [10, 20]
    }

    lrct_searcher = GridSearchCV(
        LRCTree(),
        lrct_params,
        n_jobs = -1,
        verbose = 3,
        cv = ps
    ).fit(x_train_val, y_train_val)

    model_report(lrct_searcher, x_test, y_test)
