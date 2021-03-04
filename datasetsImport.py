# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 04:17:11 2017

@author: João
"""

#Importação dos datasets
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from scipy.stats import mode
import numpy as np

dataIris = load_iris()

dataBreastCancer = load_breast_cancer()

dataDiabetes = load_diabetes()


data = dataIris.data

#data = dataBreastCancer.data

#data = dataDiabetes.data

kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

clusters = kmeans.fit_predict(data)

print(clusters)

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(dataIris.target[mask])[0]

y_true = dataIris.target

y_pred = labels

print(f1_score(y_true, y_pred, average='macro'))

print(accuracy_score(y_true, y_pred))