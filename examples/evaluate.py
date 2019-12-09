# 元データと低次元表現それぞれに対し精度を計算する
# X:特徴行列, L: ラベル

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

def kNN_acc(X, L):
    X_train, X_test, Y_train, Y_test = train_test_split(X, L, random_state=0)
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, Y_train)
    Y_pred = knc.predict(X_test)
    score = knc.score(X_test, Y_test)

    return score

def visualize(X, L, cmap='Spectral', s=10):

    sns.set(context="paper", style="white")

    fig, ax = plt.subplots(figsize=(12, 10))
    color = L.astype(int)
    plt.scatter(
        X[:, 0], X[:, 1], c=color, cmap=cmap, s=s
    )
    plt.setp(ax, xticks=[], yticks=[])
    # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

    plt.show()

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

data = 'MNIST'
datasize = str(70000)
hub_org = 'hub'
iter = str(10)

file_pass = 'embed_' + hub_org + '_' + data + datasize + '_' + iter + '.npz'

npz = np.load(file_pass)
X = npz['X']
L = npz['L']
emb = npz['emb']

result_knn = []
result_acc = []
result_ari = []
result_ami = []

for i, e in enumerate(emb):

    knn_acc = kNN_acc(e, L)
    acc, ari, ami = kmeans_acc_ari_ami(e, L)

    result_knn.append(knn_acc)
    result_acc.append(acc)
    result_ari.append(ari)
    result_ami.append(ami)

result = np.array((result_knn, result_acc, result_ari, result_ami))
# with open('examples/result_org_'+data+datasize+'.csv', 'w') as f:
np.savetxt('result_' + hub_org + '_' + data + datasize + '_' + iter + '.txt', result)

# 統計処理
file_pass = 'result_' + hub_org + '_' + data + datasize + '_' + iter + '.txt'
result_lst = np.loadtxt(file_pass)
results = DataFrame()
results['knn'] = result_lst[0]
results['acc'] = result_lst[1]
results['ari'] = result_lst[2]
results['ami'] = result_lst[3]
# descriptive stats
print(results.describe())