import numpy as np
import matplotlib.pyplot as plt
import umap
import Macosko_utils as utils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from evaluate import kNN_acc, kmeans_acc_ari_ami
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from munkres import Munkres
import csv
from numpy import savetxt
from pandas import DataFrame
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages
from MantelTest import Mantel
from hub_toolbox.distances import euclidean_distance
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numba


def kNN_acc(X, L):
    X_train, X_test, Y_train, Y_test = train_test_split(X, L, random_state=0)
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, Y_train)
    Y_pred = knc.predict(X_test)
    score = knc.score(X_test, Y_test)

    return score


def kmeans_acc_ari_ami(X, L):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    n_clusters = len(np.unique(L))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)

    y_pred = kmeans.fit_predict(X)
    y_pred = y_pred.astype(np.int64)
    y_true = L.astype(np.int64)
    assert y_pred.size == y_true.size

    y_pred = y_pred.reshape((1, -1))
    y_true = y_true.reshape((1, -1))

    # D = max(y_pred.max(), L.max()) + 1
    # w = np.zeros((D, D), dtype=np.int64)
    # for i in range(y_pred.size):
    #     w[y_pred[i], L[i]] += 1
    # # from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(w.max() - w)
    #
    # return sum([w[i, j] for i in row_ind for j in col_ind]) * 1.0 / y_pred.size

    if len(np.unique(y_pred)) == len(np.unique(y_true)):
        C = len(np.unique(y_true))

        cost_m = np.zeros((C, C), dtype=float)
        for i in np.arange(0, C):
            a = np.where(y_pred == i)
            # print(a.shape)
            a = a[1]
            l = len(a)
            for j in np.arange(0, C):
                yj = np.ones((1, l)).reshape(1, l)
                yj = j * yj
                cost_m[i, j] = np.count_nonzero(yj - y_true[0, a])

        mk = Munkres()
        best_map = mk.compute(cost_m)

        (_, h) = y_pred.shape
        for i in np.arange(0, h):
            c = y_pred[0, i]
            v = best_map[c]
            v = v[1]
            y_pred[0, i] = v

        acc = 1 - (np.count_nonzero(y_pred - y_true) / h)

    else:
        acc = 0
    # print(y_pred.shape)
    y_pred = y_pred[0]
    y_true = y_true[0]
    ari, ami = adjusted_rand_score(y_true, y_pred), adjusted_mutual_info_score(y_true, y_pred)

    return acc, ari, ami

@numba.jit()
def mantel_test(X, L, embed, describe = True):
    sss = StratifiedShuffleSplit(n_splits=50, test_size=1000, random_state=0)
    sss.get_n_splits(X, L)

    label_type = list(set(L))
    r_lst = np.array([])
    p_lst = np.array([])
    for _, idx in sss.split(X, L):
        # print('Index: ', idx)
        # X_test = X[idx]
        # y_train =

        X_high, L_hl = X[idx], L[idx]
        X_low = embed[idx]

        # print(X_high.shape, L_high.shape)
        # print(X_low.shape, L_low.shape)

        label_idx = []

        for _, i in enumerate(label_type):
            l_idx = np.where(L_hl == i)
            label_idx.append(l_idx)

        # print(label_type)

        # label_idx
        X_high_lst = []
        X_low_lst = []
        # for _, i in enumerate(label_type):
        #     X_high_lst.append(X_high[label_idx[i]])
        for i, _ in enumerate(label_type):
            centroid = np.mean(X_high[label_idx[i]], axis=0)
            # print(centroid)
            X_high_lst.append(centroid)
            # print(centroid.shape)
            # X_high_lst.append((X_high[label_idx[i]] - centroid))
            # X_high_lst[label_idx[i]] = np.sqrt(np.linalg.norm(X_high[label_idx[i]] - centroid, ord=2))
            # for _, i in enumerate(label_type):

            centroid = np.mean(X_low[label_idx[i]], axis=0)
            X_low_lst.append(centroid)
            # print(centroid.shape)
            # X_high_lst.append((X_low[label_idx[i]] - centroid))
            # X_low_lst[label_idx[i]] = np.sqrt(np.linalg.norm(X_low[label_idx[i]] - centroid, ord=2))

        # print(X_low_lst[0].shape, centroid.shape)
        D_high = euclidean_distance(X_high_lst)
        D_low = euclidean_distance(X_low_lst)
    # print(D_high, D_low)

        r, p, z = Mantel.test(D_high, D_low, perms=1000, method='pearson', tail='upper')
        r_lst = np.append(r_lst, r)
        p_lst = np.append(p_lst, p)

    if describe == True:
        print(pd.DataFrame(pd.Series(r_lst.ravel()).describe()).transpose())

    return r_lst, p_lst

npz = np.load('macosko2015.npz')
X = npz['X']
n = 10000
X = X[:n]
print(X.shape)
cell_type2 = npz['cell_type2']
y = npz["cell_type1"].astype(str)
y = y[:n]
L = [int(float(x)) for x in cell_type2]
L = np.array(L)
L = L[:n]

# U, S, V = np.linalg.svd(X, full_matrices=False)
# U[:, np.sum(V, axis=1) < 0] *= -1
# X = np.dot(U, np.diag(S))
# X = X[:, np.argsort(S)[::-1]][:, :100]
#
# print(X.shape)

import umap
# emb = umap.UMAP(metric='precomputed').fit_transform(X)
# reducer = umap.UMAP()
reducer = umap.UMAP(metric='precomputed')
emb = reducer.fit_transform(X)
# emb = reducer.fit_transform(X_reduced)

utils.plot(emb, y, colors=utils.MACOSKO_COLORS)
plt.show()

kNN_score = kNN_acc(emb, L)
acc, ari, ami = kmeans_acc_ari_ami(emb, L)
pcc, _ = mantel_test(X, L, emb)

print("kNN:", kNN_score, "acc:", acc, "ari:", ari, "ami:", ami)

#    count      mean       std       min       25%       50%       75%       max
# 0   50.0  0.694691  0.054863  0.517372  0.665805  0.708712  0.728032  0.791418
# kNN: 0.45 acc: 0.12280000000000002 ari: 0.014226538949639137 ami: 0.10134672117996671
