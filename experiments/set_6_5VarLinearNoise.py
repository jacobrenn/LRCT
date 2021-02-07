# Single linear split across two variables, 5 total variables present

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
    # Generate the data, domain for x0 is [0, 14], domain for x1 is [0, 8]
    np.random.seed(4736)
    two_var_x = np.random.random((1000, 5))
    two_var_x[:, 0] = two_var_x[:, 0] * 14
    two_var_x[:, 1] = two_var_x[:, 1] * 8
    two_var_x[:, 2] = two_var_x[:, 2] * -6
    two_var_x[:, 3] = two_var_x[:, 3] * 3
    two_var_x[:, 4] = two_var_x[:, 4] * -5
    target = (two_var_x[:, 1] > -1 * two_var_x[:, 0] + 10).astype(int)

    # Create train, val, test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(two_var_x, target, test_size = 0.3, random_state = 134456)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.4, random_state = 89582)

    for prop in [0.1, 0.2, 0.3]:
        selected_indices = np.random.choice(np.arange(y_train.shape[0]), np.round(y_train.shape[0] * prop).astype(int), replace = False)
        y_train_exp = y_train.copy()
        y_train_exp[selected_indices] = 1 - y_train_exp[selected_indices]
    
        # Plot the data
        plt.figure(figsize = (10, 4))
        plt.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c = y_train_exp,
            cmap = 'Set1'
        )
        plt.title(f'Training Data, Experiment 6, Noise Level {prop}', fontsize = 'xx-large')
        plt.xlabel('Feature 1', fontsize = 'x-large')
        plt.ylabel('Feature 2', fontsize = 'x-large')
        plt.xticks(fontsize = 'large')
        plt.yticks(fontsize = 'large')
        plt.xlim(-0.5, 14.5)
        plt.ylim(-0.5, 8.5)
        plt.savefig(f'exp6/exp_6_training_{prop}.png')

        # Train the models
        lrct = LRCTree(max_depth = 1).fit(x_train, y_train_exp)
        cart = DecisionTreeClassifier(max_depth = 1).fit(x_train, y_train_exp)
        oc1 = ObliqueTree(splitter = 'oc1')
        oc1.fit(x_train, y_train_exp)

        # Print results
        print(f'Results for Noise Level {prop}')
        
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
        try:
            intercept = lrct.nodes[0].split['split_value']
            coef = float(lrct.nodes[0].split['coefs'][0])
            col1 = lrct.nodes[0].split['indices'][0]
            col2 = lrct.nodes[0].split['indices'][1]
            split_line = coef * x_test[:, col1] - intercept
            
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
            plt.title(f'LRCT Learned Decision Boundary, Experiment 6, Noise Level {prop}', fontsize = 'xx-large')
            plt.xlabel('Feature 1', fontsize = 'x-large')
            plt.ylabel('Feature 2', fontsize = 'x-large')
            plt.xticks(fontsize = 'large')
            plt.yticks(fontsize = 'large')
            plt.xlim(-0.5, 14.5)
            plt.ylim(-0.5, 8.5)
            plt.savefig(f'exp6/exp6_learned_{prop}.png')
        except:
            pass
