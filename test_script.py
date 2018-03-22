# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:07:15 2018

@author: Romain G. et Rémi J.
"""
"""

if t >= k use kmeans++
else kmeans||

"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from seeding import*

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)
#data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print(name)
    print('Time : ', (time() - t0))
    print('Inertia : ', estimator.inertia_)
    print('')

print('_____init: KMEANS++ SKLEARN______')
bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

print('_____init: RANDOM______')
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

print('_____init: KMEANS||______')
t0 = time()
seeding = seeding_parallel(data, 5, 30, n_digits, 'euclidean', np.random.RandomState(), weights=None)
print('Init time : ', (time() - t0))

bench_k_means(KMeans(init=seeding, n_clusters=n_digits, n_init=1),
              name="k-means||", data=data)

reduced_data = PCA(n_components=2).fit_transform(data)
seeding = seeding_parallel(reduced_data, 5, 30, n_digits, 'euclidean', np.random.RandomState(), weights=None)
for i in ['k-means++', 'random', seeding]:
    kmeans = KMeans(init=seeding, n_clusters=n_digits, n_init=1)
    Y = kmeans.fit_predict(reduced_data)
    c = ['b','g','r','m','y','k','c','0.25', '0.5', '0.75']
    plt.figure()
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color=[c[j] for j in Y])
    C = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)
    plt.scatter(C[:, 0], C[:, 1], s=140,  marker="*")
    plt.title(i)
    plt.show()

