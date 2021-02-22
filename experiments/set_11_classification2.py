# Checkerboard configuration experiment, 2 variables

# 50% of data for training, 20% for validation, 30% for testing
# Model types to choose: LRCT, OC1, CART, KNN, LogisticRegression, Fully-Connected Neural Network

# Report: accuracy, confusion_matrix, per-class and average F1, AUC

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
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
    multivar_x, target = make_classification(
        n_samples = 10000,
        n_features = 12,
        n_informative = 6,
        n_redundant = 4,
        n_repeated = 2,
        n_classes = 4,
        n_clusters_per_class = 3,
        random_state = 8312
    )

    # Create train, val, test sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(multivar_x, target, test_size = 0.3, random_state = 133523)
    val_indices = np.random.choice(x_train_val.shape[0], int(x_train_val.shape[0] * 0.4), replace = False)
    train_val_fold = [0 if i in val_indices else -1 for i in range(x_train_val.shape[0])]
    ps = PredefinedSplit(test_fold = train_val_fold)

    x_train = x_train_val[np.array(train_val_fold) == -1, :]
    y_train = y_train_val[np.array(train_val_fold) == -1].astype(int)

    x_val = x_train_val[np.array(train_val_fold) != -1, :]
    y_val = y_train_val[np.array(train_val_fold) != -1].astype(int)
    
    # OC1 First
    
    oc1_model = ObliqueTree(splitter = 'oc1')
    oc1_model.fit(x_train, y_train)

    print('OC1 Model Performance:')
    print('\n')
    
    model_report(oc1_model, x_test, y_test, binary = False)
    print('\n\n')

    # CART
    cart_model = DecisionTreeClassifier()
    cart_params = {
        'max_depth' : range(1, 12),
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
    model_report(cart_searcher, x_test, y_test, binary = False)
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
    model_report(knn_searcher, x_test, y_test, binary = False)
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
    model_report(log_reg_searcher, x_test, y_test, binary = False)
    print('\n\n')

    # Neural Network
    input_layer = tf.keras.layers.Input(x_train.shape[1])
    x = tf.keras.layers.Dense(100, activation = 'relu')(input_layer)
    x = tf.keras.layers.Dense(100, activation = 'relu')(x)
    x = tf.keras.layers.Dense(100, activation = 'relu')(x)
    x = tf.keras.layers.Dense(100, activation = 'relu')(x)
    x = tf.keras.layers.Dense(100, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(4, activation = 'softmax')(x)
    neural_net = tf.keras.models.Model(input_layer, output_layer)
    neural_net.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

    checkpoint_file = '/tmp/checkpoint'
    cb = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_file,
        save_weights_only = True,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = True
    )

    neural_net.fit(
        x_train,
        y_train,
        epochs = 1000,
        callbacks = [cb],
        validation_data = (x_val, y_val),
        verbose = 0
    )
    neural_net.load_weights(checkpoint_file)

    print('Neural Network Model Performance:')
    print('\n')
    model_report(neural_net, x_test, y_test, neural_net = True, binary = False)
        
    
    # Lastly, the LRCT
    lrct = LRCTree()
    lrct_params = {
        'method' : ['ols', 'ridge', 'lasso'],
        'n_independent' : range(1, 3),
        'max_depth': range(3, 5),
        'min_samples_split': [5],
        'min_samples_leaf': [5],
        'highest_degree': range(1, 3),
        'n_bins': [10]
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
    model_report(lrct_searcher, x_test, y_test, binary = False)
