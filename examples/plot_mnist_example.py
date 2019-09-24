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
from utils import global_score
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

sns.set(context="paper", style="white")

# (X, L), (_, _) = keras.datasets.fashion_mnist.load_data()
#
# n = 60000
# # print(X.shape)
# X = X[:n].reshape((n, 28*28))
# X = X / 255.
# L = L[:n].astype(int)

mnist = fetch_openml('mnist_784', version=1)
n = 70000
X = mnist.data[:n]
L = mnist.target[:n].astype(int)


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

start = time.time()

# reducer = umap.UMAP(metric='precomputed', random_state=42)
# neigbour_graph = kneighbors_graph(X, algorithm='hnsw', algorithm_params={'n_candidates': 100}, n_neighbors=15, mode='distance', hubness='mutual_proximity',
#                                           hubness_params={'method': 'normal'})
# embedding = reducer.fit_transform(neigbour_graph)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

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

fig, ax = plt.subplots(figsize=(12, 10))
color = L.astype(int)
plt.scatter(
    embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=1
)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()

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