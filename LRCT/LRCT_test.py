from LRCT import LRCTree

import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        'x1' : range(100),
        'x2' : range(100)
    }
)

y = np.array([0]*50 + [1]*50)

tree = LRCTree()
tree = tree.fit(df, y)
tree.describe()