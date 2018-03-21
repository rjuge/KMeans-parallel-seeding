# coding: utf-8

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import os
import multiprocessing as mp

class Sampler(object):
    def __init__(self, C, l, distance, random_state):
        self.C = C
        self.l = l
        self.distance = distance
        self.random_state = random_state
    def __call__(self, x):
        d = dist(x, self.C, metric_=self.distance)**2
        s = self.random_state.choice(x.shape[0], p=min(1, self.l*d)) # TODO: access sum
        if(s==1):
            return x

def dist(X, Y, metric_):
    return pairwise_distances(X, Y, metric=metric_)

def overseeding(X, t, l, distance, random_state, weights):
    # initial center sampled randomly
    row = random_state.choice(X.shape[0], p=weights/weights.sum())
    C = np.expand_dims(X[row, :], axis=0)
    # compute probabilities
    di = dist(X, C, metric_=distance)
    # normalise
    q = np.minimum(np.ones(di.shape), di)
    q = np.squeeze(q/np.sum(q))
    # multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    X_list = [np.expand_dims(X[i,:], axis=0) for i in range(X.shape[0])]
    for i in range(t):
        # run pool
        #cand_id = random_state.choice(X.shape[0], size=np.int(l*X.shape[0]/100), p=q).astype(np.intp)
        sampler = Sampler(C, l, distance, random_state)
        cand_id = pool.map(sampler, X_list) # TODO: check output
        pool.close()
        pool.join()
        # extract selected points
        C_tmp = X[cand_id]
        C = np.vstack([C, C_tmp])
        # update proba
        di = np.min(dist(X, C, metric_=distance), axis=1)
        # normalise
        q = np.minimum(np.ones(di.shape), di)
        q = np.squeeze(q/np.sum(q))
    return C

def seeding_parallel(X, t, l, K, distance, random_state, weights=None):
    if weights is None:
        weights = np.ones(X.shape[0])
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    # overseeding step
    B = overseeding(X, t, l, distance, random_state, weights)
    d = dist(X, B, metric_=distance)
    # assign to clusters
    idx = np.argmin(d, axis=1)
    wc = np.ones(B.shape[0])
    # compute weights
    for i in range(wc.shape[0]):
        wc[i] = idx.tolist().count(i)
    # seeding++ step
    C = seeding_plusplus(B, np.expand_dims(wc, axis=1), K, distance, random_state)
    return C

def seeding_plusplus(X, wc, K, distance, random_state):
    # init first center
    row = random_state.choice(X.shape[0], p=(wc/wc.sum())[:,0])
    C = np.expand_dims(X[row, :].T, axis=0)
    # compute proba
    di = np.multiply(dist(X, C, metric_=distance), wc)
    q = np.squeeze(di/np.sum(di))
    for i in range(1, K):
        # sample relevant samples
        cand_id = random_state.choice(X.shape[0], p=q)
        C = np.vstack([C, X[cand_id, :]])
        # update proba
        di = np.amin(np.multiply(dist(X, C, metric_=distance), wc), axis=1)
        q = di/np.sum(di)
    return C
