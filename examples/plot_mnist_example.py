"""
UMAP on the MNIST Digits dataset
--------------------------------

A simple example demonstrating how to use UMAP on a larger
dataset such as MNIST. We first pull the MNIST dataset and
then use UMAP to reduce it to only 2-dimensions for
easy visualisation.

Note that UMAP manages to both group the individual digit
classes, but also to retain the overall global structure
among the different digit classes -- keeping 1 far from
0, and grouping triplets of 3,5,8 and 4,7,9 which can
blend into one another in some cases.
"""
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import hub_toolbox
from hub_toolbox.distances import euclidean_distance
from utils import calculate_AUC
from utils import global_score, mantel_test
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

# sns.set(context="paper", style="white")
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

# dataset = 'F-MNIST'
dataset = 'MNIST'
# dataset = 'CIFAR10'

if dataset == 'F-MNIST':
    (X, L), (X_test, L_test) = keras.datasets.fashion_mnist.load_data()

    n = 70000
    # print(X.shape)
    # X = X[:n].reshape((n, 28 * 28))
    X = X.reshape((X.shape[0], 28 * 28))
    X_test = X_test.reshape((X_test.shape[0], 28 * 28))
    X = np.vstack((X, X_test))
    X = X[:n]
    X = X / 255.
    L = np.hstack((L, L_test))
    L = L.astype(int)

    L = L[:n]


elif dataset == 'MNIST':
    mnist = fetch_openml('mnist_784', version=1)
    n = 70000
    X = mnist.data[:n]
    X = X / 255.
    L = mnist.target[:n].astype(int)

elif dataset == "CIFAR10":
    (X, L), (X_test, L_test) = keras.datasets.cifar10.load_data()

    n = 10000
    X = X.reshape((X.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    X = np.vstack((X, X_test))
    X = X[:n]
    X = X / 255.
    L = np.vstack((L, L_test))
    L = L.flatten()
    L = L.astype(int)



    L = L[:n]

if dataset == 'F-MNIST':
    k = 5
    min_dist = 0.1
elif dataset == 'MNIST' or dataset == 'CIFAR10':
    k = 10
    min_dist = 0.001

emb_org_list = []
emb_hub_list = []

result_hub = []
result_org = []

iter = 10

# seed = 42
# reducer = umap.UMAP(n_neighbors=k, min_dist=min_dist, random_state=seed)
# embedding_org = reducer.fit_transform(X)

# neigbour_graph = kneighbors_graph(X, algorithm='hnsw', algorithm_params={'n_candidates': 100}, n_neighbors=k, mode='distance', hubness='mutual_proximity',
#                                           hubness_params={'method': 'normal'})
# knn_indices = neigbour_graph.indices.astype(int).reshape((X.shape[0], k))
# knn_dists = neigbour_graph.data.reshape((X.shape[0], k))
#
# for i in range(X.shape[0]):
#     for j in range(k):
#         knn_dists[i, j] = scieuc(X[i, :], X[knn_indices[i, j], :])

# reducer = umap.UMAP(metric='precomputed', n_neighbors=k, min_dist=min_dist, random_state=seed)
# embedding_org = reducer.fit_transform(X)

# embedding_org = reducer.fit_transform((knn_indices, knn_dists))


# reducer = umap.UMAP(metric='precomputed', n_neighbors=k, min_dist=min_dist, random_state=seed)
# D = euclidean_distance(X)
# D_mp = hub_toolbox.global_scaling.mutual_proximity_empiric(D=D, metric='distance')
# embedding_org = reducer.fit_transform(D_mp)


# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embedding_org[:, 0], embedding_org[:, 1], c=color, cmap="Spectral", s=10
# )
# plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()

# X_train, X_test, Y_train, Y_test = train_test_split(embedding_org, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# print("1-NN: ", score)



# mpg1 = SuQHR(hr_algorithm='mpg',  n_samples=30)
# X_sample, _, idx, _ = mpg1._random_sampling(X)
# D = cdist(X, X_sample, 'euclidean')
#
#
# def mutual_proximity_gaussi_sample(D: np.ndarray, idx: np.ndarray,
#                                    metric: str = 'distance', test_set_ind: np.ndarray = None, verbose: int = 0):
#     """Transform a distance matrix with Mutual Proximity (empiric distribution).
#
#     NOTE: this docstring does not yet fully reflect the properties of this
#     proof-of-concept function!
#
#     Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix using
#     the empiric data distribution (EXACT, rather SLOW). The resulting
#     secondary distance/similarity matrix should show lower hubness.
#
#     Parameters
#     ----------
#     D : ndarray
#         The ``n x s`` distance or similarity matrix, where ``n`` and ``s``
#         are the dataset and sample size, respectively.
#     idx : ndarray
#         The index array that determines, to which data points the columns in
#         `D` correspond.
#     metric : {'distance', 'similarity'}, optional (default: 'distance')
#         Define, whether matrix `D` is a distance or similarity matrix.
#     test_set_ind : ndarray, optional (default: None)
#         Define data points to be hold out as part of a test set. Can be:
#         - None : Rescale all distances
#         - ndarray : Hold out points indexed in this array as test set.
#     verbose : int, optional (default: 0)
#         Increasing level of output (progress report).
#     Returns
#     -------
#     D_mp : ndarray
#         Secondary distance MP empiric matrix.
#     References
#     ----------
#     .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
#            Local and global scaling reduce hubs in space. The Journal of Machine
#            Learning Research, 13(1), 2871–2902.
#     """
#     # Initialization and checking input
#     log = ConsoleLogging()
#     io.check_sample_shape_fits(D, idx)
#     io.check_valid_metric_parameter(metric)
#     n = D.shape[0]
#     s = D.shape[1]
#     j = np.ones(n, int)
#     j *= (n + 1)  # illegal indices will throw index out of bounds error
#     j[idx] = np.arange(s)
#     if metric == 'similarity':
#         self_value = 1
#     else:  # metric == 'distance':
#         self_value = 0
#     exclude_value = np.nan
#     if test_set_ind is None:
#         n_ind = range(n)
#     else:
#         n_ind = test_set_ind
#
#     # Start MP
#     D = D.copy()
#
#     if issparse(D):
#         raise NotImplementedError
#
#     # ensure correct self distances (NOT done for sparse matrices!)
#     for j, sample in enumerate(idx):
#         D[sample, j] = exclude_value
#
#     # Calculate mean and std per row, w/o self values (nan)
#     mu = np.nanmean(D, 1)
#     sd = np.nanstd(D, 1, ddof=0)
#     # Avoid downstream div/0 errors
#     sd[sd == 0] = 1e-7
#
#     # set self dist/sim back to self_value to avoid scipy warnings
#     for j, i in enumerate(idx):
#         D[i, j] = self_value
#
#     # # MP Gaussi
#     # D_mp = np.zeros_like(D)
#     # for sample, i in enumerate(n_ind):
#     #     if verbose and ((i + 1) % 1000 == 0 or i + 1 == n):
#     #         log.message("MP_gaussi: {} of {}.".format(i + 1, n), flush=True)
#     #     j = slice(0, s)
#     #
#     #     if metric == 'similarity':
#     #         p1 = norm.cdf(D[i, j], mu[i], sd[i])
#     #         p2 = norm.cdf(D[i, j], mu[idx], sd[idx])
#     #         D_mp[i, j] = (p1 * p2).ravel()
#     #     else:
#     #         # Survival function: sf(.) := 1 - cdf(.)
#     #         p1 = norm.sf(D[i, j], mu[i], sd[i])
#     #         p2 = norm.sf(D[i, j], mu[idx], sd[idx])
#     #         D_mp[i, j] = (1 - p1 * p2).ravel()
#     #
#     # # Ensure correct self distances
#     # for j, sample in enumerate(idx):
#     #     D_mp[sample, j] = self_value
#
#     # if test_set_ind is None:
#     #     return D_mp
#     # else:
#     #     return D_mp[test_set_ind]
#
#     return mu, sd
#
# mu, sd = mutual_proximity_gaussi_sample(D, idx)
#
# reducer = umap.UMAP(metric_knn='mpg', metric_knn_kwds={"mu":mu, "sd":sd}, random_state=42)
# embedding = reducer.fit_transform(X)

# D, labels, vectors = hub_toolbox.io.load_dexter()
# L = labels


# reducer = umap.UMAP(metric='precomputed', random_state=42)
# D = euclidean_distance(X)
# D_mp = hub_toolbox.global_scaling.mutual_proximity_gaussi(D=D, metric='distance', sample_size=100)
# embedding = reducer.fit_transform(D_mp)

# start = time.time()
for i in range(iter):
    seed = random.randint(1, 100)
    # seed = 42

    # reducer = umap.UMAP(n_neighbors=k, min_dist=min_dist, metric='precomputed', random_state=seed)
    # neigbour_graph = kneighbors_graph(X, algorithm='hnsw', algorithm_params={'n_candidates': 100}, n_neighbors=k, mode='distance', hubness='mutual_proximity',
    #                                       hubness_params={'method': 'normal'})
    # embedding_hub = reducer.fit_transform(neigbour_graph)
    start1 = time.time()
    # D = euclidean_distance(X)
    reducer = umap.UMAP(metric='precomputed', n_neighbors=k, min_dist=min_dist, random_state=seed)
    embedding_hub = reducer.fit_transform(X)
    elapsed_time1 = time.time() - start1

    start2 = time.time()
    reducer = umap.UMAP(n_neighbors=k, min_dist=min_dist, random_state=seed)
    embedding_org = reducer.fit_transform(X)
    elapsed_time2 = time.time() - start2

    emb_org_list.append(embedding_org)
    emb_hub_list.append(embedding_hub)
    print('hub: ', elapsed_time1)
    print('org: ', elapsed_time2)

#     # r_lst_org = mantel_test(X, L, embedding_org)
#     # r_lst_hub = mantel_test(X, L, embedding_hub)
#     #
#     # fig = plt.figure()
#     # ax = fig.add_subplot(1, 1, 1)
#     # ax.boxplot([r_lst_org, r_lst_hub], labels=['original', 'with HR'])
#     # # ax.set_xlabel('methods')
#     # ax.set_ylabel('PCC')
#     # ax.set_ylim(0.6, 1)
#     #
#     # plt.show()
#
#
# # elapsed_time = time.time() - start
# # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
#     fig, ax = plt.subplots(figsize=(12, 10))
#     color = L.astype(int)
#     plt.scatter(
#         embedding_org[:, 0], embedding_org[:, 1], c=color, cmap="Spectral", s=10
#     )
#     plt.setp(ax, xticks=[], yticks=[])
#     # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
#     plt.show()
#
#     fig, ax = plt.subplots(figsize=(12, 10))
#     color = L.astype(int)
#     plt.scatter(
#         embedding_hub[:, 0], embedding_hub[:, 1], c=color, cmap="Spectral", s=10
#     )
#     plt.setp(ax, xticks=[], yticks=[])
#     # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
#     plt.show()
#
# # 1-NN
#     X_train, X_test, Y_train, Y_test = train_test_split(embedding_hub, L, random_state=0)
#     knc = KNeighborsClassifier(n_neighbors=1)
#     knc.fit(X_train, Y_train)
#     Y_pred = knc.predict(X_test)
#     score = knc.score(X_test, Y_test)
# # print("1-NN: ", score)
#     result_hub.append(score)
#
#     X_train, X_test, Y_train, Y_test = train_test_split(embedding_org, L, random_state=0)
#     knc = KNeighborsClassifier(n_neighbors=1)
#     knc.fit(X_train, Y_train)
#     Y_pred = knc.predict(X_test)
#     score = knc.score(X_test, Y_test)
#     # print("1-NN: ", score)
#     result_org.append(score)
#
#     print(result_org, result_hub)

# savetxt('result_hub_smallcsv', result_org)

# reducer = umap.UMAP(metric='precomputed', random_state=42)
# sss = SuQHR(hr_algorithm = 'mpg')
# embed = sss.fit_transform(X)
# mu = sss.mu_train_
# sd = sss.sd_train_
# D = np.zeros((X.shape[0], X.shape[0]))
# for i in range(X.shape[0]):
#     p1 = norm.sf(D[i, j], mu[i], sd[i])
#     p2 = norm.sf(D[i, j], mu[j_mom], sd[j_mom])
#     D_mp[i, j] = (1 - p1 * p2).ravel()

# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=1
# )
# plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()

# # AUC
# auc = calculate_AUC(X, embedding)
# print("AUC: ", auc)

# Global Score
#
# GS = global_score(X, embedding)
# print("Global score: ", GS)
#
# # 1-NN
# X_train, X_test, Y_train, Y_test = train_test_split(embedding, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# print("1-NN: ", score)

# # load results file
# results = DataFrame()
# results['Hub'] = read_csv('result_hub_small.csv', header=None).values[:, 0]
# results['Org'] = read_csv('result_org_small.csv', header=None).values[:, 0]
# # descriptive stats
# print(results.describe())


# # box and whisker plot
# results.boxplot()
# plt.show()
# # histogram
# results.hist()
# plt.show()
#
# # p-value
# # load results1
# result1 = read_csv('result_org_FM.csv', header=None)
# values1 = result1.values[:,0]
# # load results2
# result2 = read_csv('result_hub_FM.csv', header=None)
# values2 = result2.values[:,0]
# # calculate the significance
# value, pvalue = ttest_ind(values1, values2, equal_var=True)
# print(value, pvalue)
# if pvalue > 0.05:
# 	print('Samples are likely drawn from the same distributions (fail to reject H0)')
# else:
# 	print('Samples are likely drawn from different distributions (reject H0)')


#
# # AUC:  0.1932818455849898
# # 1-NN:  0.7236
#
# # 1-NN:  0.9338666666666666
#
# # 1-NN:  0.9552
# # 1-NN:  0.718
#
# # AUC:  0.1776903621527402
# # 1-NN:  0.7236
#
# # AUC:  0.08569556227587755
# # 1-NN:  0.718
#
# # Global score:  0.9078743988020537
# # 1-NN:  0.9397333333333333


# Global score:  0.9219980769162641
# 1-NN:  0.942


# 1-NN:  0.7539333333333333
#    count      mean       std       min      25%     50%       75%       max
# 0   50.0  0.900682  0.009099  0.871956  0.89596  0.9018  0.906548  0.916395

# MNIST
# hub:  307.36033487319946
# org:  509.9469392299652

np.savez('embed_org_'+ dataset + str(n) + '_' + str(iter), X=X, L=L, emb=emb_org_list)
np.savez('embed_hub_'+ dataset + str(n) + '_' + str(iter), X=X, L=L, emb=emb_hub_list)
