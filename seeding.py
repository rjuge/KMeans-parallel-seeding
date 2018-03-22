# coding: utf-8

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def dist(X, Y, metric_):
    return pairwise_distances(X, Y, metric=metric_)**2

def overseeding(X, t, l, distance, random_state, weights):
    # initial center sampled randomly
    row = random_state.choice(X.shape[0], p=weights/weights.sum())
    C = np.expand_dims(X[row, :], axis=0)
    # compute probabilities
    di = dist(X, C, metric_=distance)
    # normalise
    q = np.minimum(np.ones(di.shape), di/np.sum(di))
    q = np.squeeze(q/np.sum(q))
    for i in range(t):
        cand_id = random_state.choice(X.shape[0], size=np.int(l*X.shape[0]/100), p=q).astype(np.intp)
        # extract selected points
        C_tmp = X[cand_id]
        C = np.vstack([C, C_tmp])
        # update proba
        di = np.min(dist(X, C, metric_=distance), axis=1)
        # normalise
        q = np.minimum(np.ones(di.shape), di/np.sum(di))
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
    row = random_state.choice(X.shape[0], p=(wc/np.sum(wc))[:,0])
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
