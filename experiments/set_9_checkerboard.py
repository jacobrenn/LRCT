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
from exp_utils import model_report

if __name__ == '__main__':
    # Generate the data, domain for x0 and x1 is [0, 10]
    np.random.seed(4736)
    two_var_x = np.random.random((10000, 2))
    two_var_x = two_var_x * 10
    target = ((np.floor(two_var_x[:, 0]) + np.floor(two_var_x[:, 1])) % 2).astype(int)

    # Create train, val, test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(two_var_x, target, test_size = 0.3, random_state = 133523)
    val_indices = np.random.choice(x_train_val.shape[0], int(x_train_val.shape[0] * 0.4), replace = False)
    train_val_fold = [0 if i in val_indices else -1 for i in range(x_train_val.shape[0])]
    ps = PredefinedSplit(test_fold = train_val_fold)

    x_train = x_train_val[np.array(train_val_fold) == -1, :]
    y_train = y_train_val[np.array(train_val_fold) == -1].astype(int)
    
    # Plot the data
    plt.figure(figsize = (10, 4))
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c = y_train,
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

    # OC1 First
    
    oc1_model = ObliqueTree(splitter = 'oc1')
    oc1_model.fit(x_train, y_train)

    print('OC1 Model Performance:')
    print('\n')
    
    model_report(oc1_model, x_test, y_test)
    print('\n\n')

    # CART
    cart_model = DecisionTreeClassifier()
    cart_params = {
        'max_depth' : range(18, 20),
        'min_samples_split' : range(5, 1, -1),
        'min_samples_leaf' : range(5, 1, -1),
    }

    cart_searcher = GridSearchCV(
        cart_model,
        cart_params,
        n_jobs = -1,
        verbose = 1,
        cv = ps
    ).fit(x_train_val, y_train_val)

    print('CART Performance:')
    print('CART Parameters:')
    print(cart_searcher.best_params_)
    print('\n')
    model_report(cart_searcher, x_test, y_test)
    print('\n\n')

    # KNN
    knn = KNeighborsClassifier()
    knn_params = {
        'n_neighbors' : range(2, 21),
        'weights' : ['uniform', 'distance']
    }

    knn_searcher = GridSearchCV(
        knn,
        knn_params,
        n_jobs = -1,
        verbose = 1,
        cv = ps
    ).fit(x_train_val, y_train_val)

    print('KNN Model Performance:')
    print('KNN Parameters:')
    print(knn_searcher.best_params_)
    print('\n')
    model_report(knn_searcher, x_test, y_test)
    print('\n\n')

    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg_params = {
        'penalty' : ['l2', 'none']
    }
    log_reg_searcher = GridSearchCV(
        log_reg,
        log_reg_params,
        n_jobs = -1,
        verbose = 1,
        cv = ps
    ).fit(x_train_val, y_train_val)

    print('Logistic Regression Model Performance:')
    print('Logistic Regression Parameters:')
    print(log_reg_searcher.best_params_)
    print('\n')
    model_report(log_reg_searcher, x_test, y_test)
    print('\n\n')

    #TODO, add neural net here
    
    
    # Lastly, the LRCT
    lrct = LRCTree()
    lrct_params = {
        'method' : ['ols', 'ridge', 'lasso'],
        'max_depth': range(18, 22),
        'min_samples_split': range(5, 2, -1),
        'min_samples_leaf': [5],
        'highest_degree': range(1, 3),
        'n_bins': [10, 20]
    }    
    
    lrct_searcher = GridSearchCV(
        lrct,
        lrct_params,
        n_jobs = -1,
        verbose = 1,
        cv = ps
    ).fit(x_train_val, y_train_val)

    print('LRCT Model Performance:')
    print('LRCT Parameters:')
    print(lrct_searcher.best_params_)
    print('\n')
    model_report(lrct_searcher, x_test, y_test)
