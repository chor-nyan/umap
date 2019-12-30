import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import hub_toolbox
from hub_toolbox.distances import euclidean_distance
# from utils import calculate_AUC
# from utils import global_score, mantel_test
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skhubness.neighbors import kneighbors_graph
from hub_toolbox.approximate import SuQHR
import hub_toolbox
import numpy as np
from umap.utils import fast_knn_indices
import keras
from scipy.spatial.distance import cdist, squareform
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import hub_toolbox
from scipy.spatial.distance import cdist, squareform
from hub_toolbox.distances import euclidean_distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skhubness.neighbors import kneighbors_graph
from hub_toolbox.approximate import SuQHR
import hub_toolbox
import numpy as np
from umap.utils import fast_knn_indices
import keras
from functools import partial
from itertools import filterfalse
import ctypes
import numpy as np
from scipy.special import gammainc  # @UnresolvedImport
from scipy.stats import norm
from scipy.sparse import lil_matrix, csr_matrix, issparse
from multiprocessing import Pool, cpu_count, current_process
from multiprocessing.sharedctypes import Array
from hub_toolbox import io
from hub_toolbox.htlogging import ConsoleLogging
import numba
import umap.distances
import time
from numpy import savetxt
import random
from scipy.stats import normaltest
from pandas import read_csv
from scipy.stats import ttest_ind
from pandas import DataFrame
from scipy.spatial.distance import euclidean as scieuc
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

def visualize(X, L, L_int=True, cmap='Spectral', s=10):

    sns.set(context="paper", style="white")

    fig, ax = plt.subplots(figsize=(12, 10))
    if L_int:
        color = L.astype(int)
    else:
        color = L
    plt.scatter(
        X[:, 0], X[:, 1], c=color, cmap=cmap, s=s
    )
    plt.setp(ax, xticks=[], yticks=[])
    # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

    plt.show()

def save_visualization(X, L, cmap='viridis', s=0.1, dir='./fig_vis/', dataset = 'F-MNIST', hub_org = 'org', i=0):

    sns.set(context="paper", style="white")

    fig, ax = plt.subplots(figsize=(12, 10))
    color = L.astype(int)
    plt.scatter(
        X[:, 0], X[:, 1], c=color, cmap=cmap, s=s
    )
    plt.setp(ax, xticks=[], yticks=[])
    if hub_org == 'org':
        model = 'UMAP'
    else:
        model = 'HR-UMAP'
    # plt.title(dataset + " data by " + model, fontsize=18)

    # # pdfファイルの初期化
    # pp = PdfPages(dir + dataset + '_' + model + str(i+1) + '.pdf')
    #
    # # figureをセーブする
    # plt.savefig(pp, format='pdf')
    #
    # # pdfファイルをクローズする。
    # pp.close()

    plt.savefig(dir + dataset + '_' + model + str(i+1) + '.png')


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
        label_type_minibatch = label_type.copy()
        no_exist = []
        for _, i in enumerate(label_type_minibatch):
            l_idx = np.where(L_hl == i)
            if not l_idx[0].size == 0:
                label_idx.append(l_idx)
            else:
                no_exist.append(i)

        label_type_minibatch = list(set(label_type_minibatch) - set(no_exist))

        # print(label_type)

        # label_idx
        X_high_lst = []
        X_low_lst = []
        # for _, i in enumerate(label_type):
        #     X_high_lst.append(X_high[label_idx[i]])
        for i, _ in enumerate(label_type_minibatch):
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

        r, p, z = Mantel.test(D_high, D_low, perms=10000, method='pearson', tail='upper')
        r_lst = np.append(r_lst, r)
        p_lst = np.append(p_lst, p)

    if describe == True:
        print("p-value:", p_lst)
        print(pd.DataFrame(pd.Series(r_lst.ravel()).describe()).transpose())

    return r_lst, p_lst
    # # return np.mean(r_lst)
    # print(pd.DataFrame(pd.Series(r_lst.ravel()).describe()).transpose())
    # print('r: ', r, 'p: ', p, 'z: ', z)

def box_plot_PCC(r_lst_org, r_lst_hub, save=False, dir='./fig_boxplot/', dataset = 'F-MNIST', i=0):

    # sns.set()
    sns.set(context="paper")
    # colors = ['blue', 'red']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('medianprops')
    # medi_style = dict(color='b', lw=30)

    ax.boxplot([r_lst_org, r_lst_hub], patch_artist=True, labels=['UMAP', 'HR-UMAP'])

    # for b, c in zip(bp['boxes'], colors):
    #     b.set(color=c, linewidth=1)  # boxの外枠の色
    #     b.set_facecolor(c)  # boxの色

    ax.set_xlabel('Model')
    ax.set_ylabel('Pearson correlation')
    ax.set_ylim(0.5, 0.9)

    if save:
        plt.savefig(dir + dataset + '_boxplot_' + str(i+1) + '.png')
    else:
        plt.show()



npz = np.load("tasic2018_preprocessed.npz")
X = npz['X']
L = npz['L']
c = npz['c']

subdata = False
if subdata:
    n = 10000
    X = X[:n]
    L = L[:n]
    c = c[:n]

pca = False
if pca:
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :100]

print(X.shape)

emb_org_list = []
emb_hub_list = []

iter_n = 10

seed_lst = random.sample(range(100), k=iter_n)
print(seed_lst)
for i in range(iter_n):
    # visualize(emb, c, L_int=False, s=1)
    # pcc, _ = mantel_test(X, L, emb)

    seed = seed_lst[i]

    emb_hub = umap.UMAP(init="random", metric='precomputed', random_state=seed).fit_transform(X)
    emb_org = umap.UMAP(init="random", random_state=seed).fit_transform(X)

    knn_score_hub = kNN_acc(emb_hub, L)
    acc_hub, ari_hub, ami_hub = kmeans_acc_ari_ami(emb_hub, L)

    knn_score_org = kNN_acc(emb_org, L)
    acc_org, ari_org, ami_org = kmeans_acc_ari_ami(emb_org, L)
    print("kNN:", knn_score_org, knn_score_hub, "acc:", acc_org, acc_hub, "ari:", ari_org, ari_hub, "ami:", ami_org, ami_hub)

    pcc_org, pvalue_org = mantel_test(X, L, emb_org)
    pcc_hub, pvalue_hub_ = mantel_test(X, L, emb_hub)

    emb_org_list.append(emb_org)
    emb_hub_list.append(emb_hub)


np.savez('embed_org_' + "tasic2018", X=X, L=L, emb=emb_org)
np.savez('embed_hub_' + "tasic2018", X=X, L=L, emb=emb_hub)

# [84, 80, 73, 45, 11, 91, 75, 12, 64, 52]