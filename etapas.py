import time
import warnings

import numpy as np

# from sklearn.model_selection import cross_val_predict #Para heatmap
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from utils import get_score_stts

def phase_one(dataset, method, needs_discretizer=False, random_state=None):
    start = time.process_time()
    x, y = dataset.data, dataset.target

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)

    pipe_list = [('transformer', StandardScaler()), ('estimator', method)]
    if needs_discretizer: 
        discretizer = KBinsDiscretizer(n_bins=2*len(np.unique(y)), encode = 'ordinal', strategy = 'kmeans')
        pipe_list.insert(1, ('discretizer', discretizer))

    pipeline = Pipeline(pipe_list)

    with warnings.catch_warnings():
        if needs_discretizer: #Ignora warnings causados pelo número de bins
            warnings.simplefilter('ignore')
        scores = cross_val_score(pipeline, x, y, cv=rkf, scoring="accuracy")
        
    status = get_score_stts(scores)

    return {"scores": scores, "status": status, "time": time.process_time()-start}

def phase_two(dataset, method, grid, random_state=None):
    start = time.process_time()
    x, y = dataset.data, dataset.target

    pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', method)])

    inner = StratifiedKFold(n_splits=4,random_state=random_state, shuffle=True)
    inner_gs = GridSearchCV(estimator=pipeline, param_grid = grid, scoring='accuracy', cv = inner)
        
    outter = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)

    scores = cross_val_score(inner_gs, x, y, scoring='accuracy', cv = outter)
    status = get_score_stts(scores)

    return {"scores": scores, "status": status, "time": time.process_time()-start}