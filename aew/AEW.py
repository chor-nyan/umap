# INPUT
#  X: d (features) times n (instances) input data matrix
#  param: The structure variable containing the following field:
#  max_iter: The maximum number of iteration for gradient descent
#  k: The number of nearest neighbors
#  sigma: Initial width parameter setting 'median'|'local-scaling'

# OUTPUT
#  W: The optimized weighted adjacency matrix
#  W0: The initial adjacency matrix

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import fetch_openml
import numba
from numpy import linalg

mnist = fetch_openml('mnist_784', version=1)
n = 100
X = mnist.data[:n]
L = mnist.target[:n].astype(int)

@numba.jit()
def EDM(X):
    (n, _) = X.shape

    D = np.zeros((n, n), )

    # 上三角を埋めて,転置させたものを足す
    for i in np.arange(0, n - 1):
        for j in np.arange(i + 1, n):
            D[i, j] = np.linalg.norm(X[i, :] - X[j, :])  # x[i,:].shape == (784,)

    D = D + D.T

    return D


# Define KNN graph
@numba.jit()
def create_knngraph(X, k):

    n = X.shape[0]
    D = EDM(X)
    sortD = np.zeros((n, k), )
    sortD_idx = np.zeros((n, k), dtype=int)

    for i in np.arange(0, n):
        d_vec = D[i, :]  # i-th row
        v = np.argsort(d_vec)  # 昇順にソートした配列のインデックス
        sortD_idx[i, :] = v[1:k + 1]  # 距離が短い順にk個選ぶ（自分を除く）
        sortD[i, :] = d_vec[sortD_idx[i, :]]


    return sortD, sortD_idx

@numba.jit()
def initialise_W(kD, knn_idx, sigma_type='median'):
    n, k = kD.shape
    W = np.zeros((n, n))
    if sigma_type == 'median':
        sigma = np.mean(kD)
        if sigma == 0:
            sigma = 1
        for i in range(n):
            W[i, knn_idx[i, :]] = np.exp(-kD[i, :] / (2 * sigma**2))

    elif sigma_type == 'local_scaling':
        if k < 7:
            sigma = kD[:, -1]
        else:
            sigma = kD[:, 7]

        sigma[sigma == 0] = 1
        for i in range(n):
            W[i, knn_idx[i, :]] = np.exp(-kD[i, :] / (sigma[i] * sigma[knn_idx[i, :]]))

    W = np.maximum(W, W.T)

    return W, sigma




# def aew(X, param = {'ex': 0, 'tol':1e-4, 'beta':0.1, 'beta_p':0,
#                     'max_beta_p':8, 'rho':1e-3, 'k':5, 'sigma':'median',
#                     'max_iter':100}):
#
#     n, d = X.shape
#     ex = param['ex']
#     tol = param['tol']
#     beta = param['beta']
#     beta_p = param['beta_p']
#     max_beta_p = param['max_beta_p']
#     rho = param['rho']
#     max_iter = param['max_iter']
#
#     W0, sigma0 = generate_nngraph(X, param)
#     L = np.eye(d, dtype=int)
#
#     Xori = X
#     if len(sigma0) > 1:
#         dist = squareform(pdist(X) ** 2)
#         sigma0 = sigma0.reshape((n, 1))
#         dist /= sigma0 @ sigma0.T
#
#     else:
#         X /= np.sqrt(2) * sigma0
#         dist = squareform(pdist(X) ** 2)
#
#     edge_idx = np.nonzero(W0)
#     W = np.zeros(n, dtype=int)
#     W[edge_idx] = np.exp(-dist[edge_idx])
#
#     Gd = np.zeros((n, n, d), dtype=int)
#     W_idx = []
#     for i in range(n):
#         W_idx.append(np.nonzero(W[i, :]))
#         for j in W_idx[i]:
#             if W[i, j]:
#                 Gd[i, j, :] = -(X[i, :] - X[j, :]) * (X[i, :] - X[j, :])
#                 if len(sigma0) > 1:
#                     Gd[i, j, :] = Gd[i, j, :] / (sigma0[i] * sigma0[j])
#
#
#     # Simple Gradient Descent
#     d_W = np.zeros((n, n, d))
#     d_WDi = np.zeros((n, n, d))
#
#     for i in range(max_iter):
#         D = np.sum(W, axis=1)
#         for j in range(n):
#             d_W[i, W_idx[i], :] = d_W[i, W_idx[i], :] / D[i] -
#
#
#
#     return W
# # x_hat = (W @ X) / D
# # dWdsig =
# # dDdsig
# # d_W =
# # grad = X
# # grad = 1/D
# # grad = np.sum(grad, axis=0)

@numba.jit()
def calculate_grad(X, W, sigma):
    n, d = X.shape
    dWdsig = np.zeros((n, n, d))
    dDdsig = np.zeros((n, d))

    for i in range(n):
        for j in range(n):
            for k in range(d):
                dWdsig[i, j, k] = 2 * W[i, j] * (X[i, k] - X[j, k])**2 * sigma[0, k]**(-3)

    for i in range(n):
        for k in range(d):
            dDdsig[i, k] = np.sum(dWdsig[i, :, k])

    D = np.sum(W, axis=1).reshape((n, 1))
    # D = np.maximum(D, 1e-6)
    X_hat = (W @ X) / D
    grad = np.array([0.] * d).reshape((1, d))
    for p in range(d):
        for i in range(n):
            grad[0, p] += np.sum((X[i, :] - X_hat[i, :]) @ (dWdsig[i, :, p]@X - dDdsig[i, p] * X_hat[i]) / D[i])

    return grad

@numba.jit()
def obj_func(X, W, knn_idx):
    cost = 0
    n = X.shape[0]
    D = np.sum(W, axis=1)
    for i in range(n):
        cost += linalg.norm(X[i, :] - 1/D[i] * W[i, knn_idx[i, :]] @ X[knn_idx[i, :], :]) ** 2

    return cost

# main
# Initialisation
sigma_type = 'median'
kD, knn_idx = create_knngraph(X, k=5)
W, sigma0 = initialise_W(kD=kD, knn_idx=knn_idx)
sigma = np.array([sigma0] * X.shape[1]).reshape((1, X.shape[1]))
f = obj_func(X, W, knn_idx)

n = 100
x = np.linspace(0, 5, n)
np.random.seed(seed = 32)
stack = []  # プロット用のリスト

eta = 0.01  # ステップ幅

for i in range(1):
    # プロット用のリスト
    stack.append(sigma)
    # print(calculate_grad(X, W, sigma).shape)
    # パラメータ更新
    sigma = sigma - eta * calculate_grad(X, W, sigma)
    if sigma_type == 'median':
        for i in range(n):
            # W[i, knn_idx[i, :]] = np.exp(-kD[i, :] / (2 * sigma**2))
            W[i, knn_idx[i, :]] = np.exp(-np.sum((((X[i] - X[knn_idx[i]]) / sigma) ** 2) , axis=1))
    elif sigma_type == 'local_scaling':
        for i in range(n):
            W[i, knn_idx[i, :]] = np.exp(-kD[i, :] / (sigma[i] * sigma[knn_idx[i, :]]))

    # 収束判定
    if eta * linalg.norm(calculate_grad(X, W, sigma)) <= 0.0001:
        break