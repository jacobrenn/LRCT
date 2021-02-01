# Single polynomial split across two variables

# 50% of data for training, 20% for validation, 30% for testing
# Model types to choose: LRCT, OC1, CART

# Report: accuracy, confusion matrix, per-class and average F1, AUC

import numpy as np
from sklearn.model_selection import train_test_split
from LRCT import LRCTree
from sklearn_oblique_tree.oblique import ObliqueTree
from sklearn.tree import DecisionTreeClassifier
from exp_utils import model_report
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generate the data, domain for x0 is [0, 5], domain for x1 is [0, 20]
    np.random.seed(4736)
    two_var_x = np.random.random((5000, 2))
    two_var_x[:, 0] = two_var_x[:, 0] * 5
    two_var_x[:, 1] = two_var_x[:, 1] * 20
    target = (two_var_x[:, 1] > 2 + 2*two_var_x[:, 0] - two_var_x[:, 0]**2 + (two_var_x[:, 0]**3)/2).astype(int)

    # Create train, val, test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(two_var_x, target, test_size = 0.3, random_state = 134456)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.4, random_state = 89582)
        
    # Plot the data
    plt.figure(figsize = (10, 4))
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c = y_train,
        cmap = 'Set1'
    )
    plt.title('Training Data, Experiment 3', fontsize = 'xx-large')
    plt.xlabel('Feature 1', fontsize = 'x-large')
    plt.ylabel('Feature 2', fontsize = 'x-large')
    plt.xticks(fontsize = 'large')
    plt.yticks(fontsize = 'large')
    plt.xlim(-0.5, 5.5)
    plt.ylim(-0.5, 20.5)
    plt.savefig('exp3/exp_3_training.png')

    # Train the models
    lrct = LRCTree(max_depth = 1, highest_degree = 3).fit(x_train, y_train)
    cart = DecisionTreeClassifier(max_depth = 1).fit(x_train, y_train)
    oc1 = ObliqueTree(splitter = 'oc1')
    oc1.fit(x_train, y_train)

    # Print results
    print('LRCT Results')
    print('\n')
    model_report(lrct, x_test, y_test)
    print('\n\n')
    
    print('CART Results')
    print('\n')
    model_report(cart, x_test, y_test)
    print('\n\n')
    
    print('OC1 Results')
    print('\n')
    model_report(oc1, x_test, y_test)

    # Plot LRCT learned decision boundary
    intercept = lrct.nodes[0].split['split_value']
    linear_coef = float(lrct.nodes[0].split['coefs'][0])
    squared_coef = float(lrct.nodes[0].split['coefs'][1])
    cubed_coef = float(lrct.nodes[0].split['coefs'][2])
    col1 = lrct.nodes[0].split['indices'][0]
    col2 = lrct.nodes[0].split['indices'][1]
    split_line = linear_coef * x_test[:, col1] + squared_coef * x_test[:, col1]**2 + cubed_coef * x_test[:, col1]**3 - intercept

    plt.figure(figsize = (10, 4))
    plt.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c = y_test,
        cmap = 'Set1'
    )
    plt.plot(
        x_test[:, 0],
        split_line,
        c = 'black'
    )
    plt.title('LRCT Learned Decision Boundary, Experiment 3', fontsize = 'xx-large')
    plt.xlabel('Feature 1', fontsize = 'x-large')
    plt.ylabel('Feature 2', fontsize = 'x-large')
    plt.xticks(fontsize = 'large')
    plt.yticks(fontsize = 'large')
    plt.xlim(-0.5, 5.5)
    plt.ylim(-0.5, 20.5)
    plt.savefig('exp3/exp3_learned.png')
