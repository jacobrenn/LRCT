from splitting_functions import find_best_lrct_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.random((100,10)) * 1
y = (x[:,2]**2 > x[:,0]).astype(int)

x = pd.DataFrame(x, columns = [f'col_{i}' for i in range(x.shape[1])])
print(find_best_lrct_split(x, y, num_independent = 1, highest_degree = 2, method = 'lasso', n_bins = 20))
print(x.shape)

