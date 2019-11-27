# LRCT

## TODO:
   - GridSearchCV interface is currently not working (something with typechecking)
   - Refactor to do typechecking and everything in the fit method, rather than in upon initialization (as per the scikit-learn official API)
     - Still maintain @property definitions of all of the attributes
     - Do not change max_depth to inf if None is passed (do this in the fit method
