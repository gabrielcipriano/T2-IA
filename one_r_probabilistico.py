'''
Modulo da classe do classificador OneRProbabilistico.
'''

import pandas as pd
import random as rand
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

def run_roulettes(roulettes, random_state = None):
    rand.seed(random_state)
    results = []
    for attr in roulettes:
        chosen = rand.random()
        for i, odd in enumerate(attr):
            if chosen <= odd:
                results.append(i)
                break
    return results

class OneRProbabilistico(BaseEstimator):
    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state

    def fit(self, x_train, y_train):
        x_train, y_train = check_X_y(x_train, y_train)

        # Contingency Table for each attribute in x
        contin_table = [pd.crosstab(attribute, y_train) for attribute in x_train.T ]
        # index and table of the best Attribute for differentiation
        self.i_best_attr = np.argmax([table.max(axis=1).sum() for table in contin_table])
        best_attr = contin_table[self.i_best_attr]
        # class distribution
        self.class_dist = best_attr.div(best_attr.sum(axis=1), axis=0)

        return self

    def predict(self, X):
        attr_values = X.T[self.i_best_attr]
        attr_chances = self.class_dist.loc[attr_values].values

        # Build Roulette list: [0.1, 0.5, 0.4] => [0.1, 0.6, 1.0]
        roulettes = attr_chances.copy()
        for attr in roulettes:
            for a, b, i in zip(attr, attr[1:], range(1,len(attr))): attr[i] = a + b
        
        # Run roulette
        predicted_results = run_roulettes(roulettes, self.random_state)

        return predicted_results