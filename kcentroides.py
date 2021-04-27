'''
Modulo da classe do classificador KCentroides.
    Inicialização:
        O KCentroides precisa receber um K e um método de agrupamento como parâmetros.
        O método de agrupamente precisa ser no seguinte padrão:
            Parâmetros: data, k e random_seed
            Retorno: A lista de K centroides
'''

import numpy as np
from scipy.spatial.distance import cdist
import random as rand

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.cluster import KMeans

from clustering import Clustering
from genetic_algorithm import genetic

# Encapsulamento do kmeans
def kmeans(data, k, random_state=None):
    km = KMeans(n_clusters=k, random_state=random_state)
    km.fit(data)
    return km.cluster_centers_

# Encapsulamento do genetic Algorithm
def kga(data, k, random_state=None):
    rand.seed(random_state)
    problem = Clustering(data)
    centroids, _, _ =  genetic(problem, k, t_pop=10, taxa_cross=0.95, taxa_mutacao=0.2)
    return centroids


class Kcentroides(BaseEstimator):
    def __init__(self, method=None, k=5, random_state=None):
        super().__init__()
        self.k = k
        self.method = method
        self.random_state = random_state
    
    def fit(self,x_train,y_train):
        x_train,y_train = check_X_y(x_train,y_train)
        clss_names = np.unique(y_train)
        mapp = { k: i for i, k in  enumerate(clss_names)}
        

        # Faz uma lista de classes, mapeando as classes em indices
        x_classes = [[] for i in range(len(clss_names))]
        for observation, clss in zip(x_train, y_train):
            x_classes[mapp[clss]].append(observation)
        
        
        # gera os k centroids de cada classe
        c_classes = [[] for i in range(len(clss_names))]
        for clss, x_clss in enumerate(x_classes):
            c_classes[clss] = self.method(np.array(x_clss), self.k, self.random_state)

        self.__c_classes = np.asarray(c_classes)
        self.__clss_names = clss_names

        return self

    def predict(self,x_test):
        dist_array = [ np.min(cdist(x_test, c_clss, 'sqeuclidean'),axis=1) for c_clss in self.__c_classes]
        return self.__clss_names[np.argmin(dist_array, axis=0)]