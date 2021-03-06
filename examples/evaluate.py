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
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages
from MantelTest import Mantel
from hub_toolbox.distances import euclidean_distance
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numba
from sklearn import neighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def kNN_acc(X, L):
    X_train, X_test, Y_train, Y_test = train_test_split(X, L, random_state=0)
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, Y_train)
    Y_pred = knc.predict(X_test)
    score = knc.score(X_test, Y_test)

    return score

def kNN_acc_kfold(X, y, n_neighbors=1):
    """
    Returns the average 10-fold validation accuracy of a NN classifier trained on the given embeddings
    Args:
        X (np.array): feature matrix of size n x d
        y (np.array): label matrix of size n x 1
        n_neighbors (int): number of nearest neighbors to be used for inference
    Returns:
        score (float): Accuracy of the NN classifier
    """
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.average(scores)

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

        r, p, z = Mantel.test(D_high, D_low, perms=10000, method='pearson', tail='upper')
        r_lst = np.append(r_lst, r)
        p_lst = np.append(p_lst, p)

    if describe == True:
        print(p_lst)
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
    ax.set_ylim(0.2, 0.8)

    if save:
        plt.savefig(dir + dataset + '_boxplot_' + str(i+1) + '.png')
    else:
        plt.show()

# # data = 'MNIST'
# # data = 'F-MNIST'
# # data = 'coil100'
# data = 'NORB'
#
# # datasize = str(70000)
# # datasize = str(7200)
# datasize = str(48600)
# # datasize =''
# hub_org = 'org'
# # hub_org = 'hub'
# iter = str(10)
#
# # path = '/home/hino/git/umap2/examples/*hub_coil100*.npz'
# # file_lst = glob.glob(path)
# # # for f in os.listdir(path):
# # #     if os.path.isfile(os.path.join(path, f)):
# # #         file_lst.append(f)
# # print(file_lst)
# #
# # emb_lst = []
# # for i, e in enumerate(file_lst):
# #     file_path = e
# #     npz = np.load(file_path)
# #     emb = npz['emb']
# #     emb = emb.reshape((1, emb.shape[0], -1))
# #     emb_lst.append(emb)
# #
# # for i in range(len(file_lst) - 1):
# #     emb_lst[i+1] = np.vstack((emb_lst[i], emb_lst[i+1]))
# #
# # print(emb_lst[len(file_lst)-1].shape)
# #
# # X = npz['X']
# # L = npz['L']
# #
# # np.savez('embed_hub_'+ "coil100" + str(7200) + '_' + str(10), X=X, L=L, emb=emb_lst[len(file_lst)-1])
#
# # seed_lst = [42, 97, 69, 99]
# # emb_lst = []
# # for i, e in enumerate(seed_lst):
# #     file_path = "embed_hub_NORB48600_Seed:" + str(e) + ".npz"
# #     # print(file_path)
# #     npz = np.load(file_path)
# #     X = npz['X']
# #     L = npz['L']
# #     emb = npz['emb']
# #     emb_lst.append(emb)
# #
# # emb = np.vstack((emb_lst[0], emb_lst[1], emb_lst[2], emb_lst[3]))
#
#
# file_path = 'embed_' + hub_org + '_' + data + datasize + '_' + iter + '.npz'
# # file_path = "embed_org_NORB48600_Seed:42.npz"
# # file_path = 'embed_' + hub_org + "_coil100" + str(7200) + '_' + str(10) + '.npz'
# npz = np.load(file_path)
# X = npz['X']
# L = npz['L']
# emb = npz['emb']
# print(emb.shape)
#
# result_knn = []
# result_acc = []
# result_ari = []
# result_ami = []
#
# pcc_lst = []
# p_value = []
#
# for i, e in enumerate(emb):
#
#     # knn_acc = kNN_acc(e, L)
#     knn_acc = kNN_acc_kfold(e, L)
#     # acc, ari, ami = kmeans_acc_ari_ami(e, L)
#     # save_visualization(e, L, dataset=data, hub_org=hub_org, i=i)
#     # r, p = mantel_test(X, L, e)
#     # pcc_lst.append(r)
#     # p_value.append(p)
#     # print("p-value:", p_value)
#     # # visualize(e, L)
#     result_knn.append(knn_acc)
#     # result_acc.append(acc)
#     # result_ari.append(ari)
#     # result_ami.append(ami)
# results = DataFrame()
# results['knn'] = result_knn
# print(results.describe())
# # # PCC =======================
# # pcc_lst = np.array(pcc_lst)
# # np.savetxt('pcc_' + hub_org + '_' + data + '.txt', pcc_lst)
# # file_pass = 'pcc_' + hub_org + '_' + data + '.txt'
# #
# # # BOX PLOT =========================================================
# # file_path_org = 'pcc_' + "org" + '_' + data + '.txt'
# # file_path_hub = 'pcc_' + "hub" + '_' + data + '.txt'
# #
# # pcc_lst_org = np.loadtxt(file_path_org)
# # pcc_lst_hub = np.loadtxt(file_path_hub)
# # for i in range(len(pcc_lst_org)):
# #   box_plot_PCC(pcc_lst_org[i], pcc_lst_hub[i], save=False, dataset=data, i=i)
#
# # LOCAL accuracy ==========================
# # result = np.array((result_knn, result_acc, result_ari, result_ami))
# # # with open('examples/result_org_'+data+datasize+'.csv', 'w') as f:
# # np.savetxt('result_' + hub_org + '_' + data + datasize + '_' + iter + '.txt', result)
# #
# # # 統計処理
# # file_path = 'result_' + hub_org + '_' + data + datasize + '_' + iter + '.txt'
# # result_lst = np.loadtxt(file_path)
# # results = DataFrame()
# # results['knn'] = result_lst[0]
# # results['acc'] = result_lst[1]
# # results['ari'] = result_lst[2]
# # results['ami'] = result_lst[3]
# # # descriptive stats
# # print(results.describe())
#
# # 0.440667   0.291125   0.148291   0.420398