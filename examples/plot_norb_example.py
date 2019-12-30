import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import norb_loader
import os
import os, gzip, bz2
import numpy
import umap
from small_norb.smallnorb.dataset import SmallNORBDataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from evaluate import kNN_acc, kNN_acc_kfold, kmeans_acc_ari_ami, mantel_test, visualize
from openTSNE import TSNE

sns.set(style='white', rc={'figure.figsize': (12, 10)})

# dataset = SmallNORBDataset(dataset_root='norb-small')

train_data = norb_loader.norb_data('train')
train_labels = norb_loader.norb_labels('train')

# n = 20000
# m = int(n/2)
# train_data = train_data[:n]
# train_labels = train_labels[:m]

X = train_data
X = X / 255.
L = np.empty([train_labels.shape[0] * 2, ], dtype=int)
L[0::2] = train_labels
L[1::2] = train_labels

# print(X.shape, L.shape)
# #
# u = umap.UMAP(n_neighbors=30).fit_transform(X)
# plt.scatter(u[:, 0], u[:, 1], s=0.1, c=L, cmap='viridis')
# plt.show()

k = 30
# seed = 42
emb_org_list = []
emb_hub_list = []

time_org_list = []
time_hub_list = []

# result_hub = []
# result_org = []
# init='random',

# X = X[:10000]
# L = L[:10000]
iter_n = 1
seed_lst = random.sample(range(100), k=iter_n)
# seed_set = set(seed_lst)
# seed_set.discard(97)
# seed_set.discard(69)
# seed_set.discard(99)
# seed_lst = list(seed_set)
# seed_lst = [42]
pca = True
if pca:
    X -= np.mean(X, axis=0)
    X = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)

print("X shape:", X.shape)
print(seed_lst)

for i in range(iter_n):
    # seed = random.randint(1, 100)
    # seed = i + 41
    seed = seed_lst[i]
    # seed = 42

    start1 = time.time()
    reducer = umap.UMAP(metric='precomputed', n_neighbors=k, random_state=seed)
    embedding_hub = reducer.fit_transform(X)
    elapsed_time1 = time.time() - start1

    start2 = time.time()
    reducer = umap.UMAP(n_neighbors=k, random_state=seed)
    embedding_org = reducer.fit_transform(X)
    elapsed_time2 = time.time() - start2

    start3 = time.time()
    embedding_TSNE = TSNE().fit(X)
    elapsed_time3 = time.time() - start3

    emb_org_list.append(embedding_org)
    emb_hub_list.append(embedding_hub)

    time_org_list.append(elapsed_time2)
    time_hub_list.append(elapsed_time1)
    print('org: ', elapsed_time2)
    print('hub: ', elapsed_time1)
    print('TSNE:', elapsed_time3)

time_org = np.array(time_org_list)
time_hub = np.array(time_hub_list)

mean_time_org = np.mean(time_org)
mean_time_hub = np.mean(time_hub)

print("org:", mean_time_org, "hub:", mean_time_hub)

kNN_score_org = kNN_acc_kfold(embedding_org, L)
kNN_score_hub = kNN_acc_kfold(embedding_hub, L)
kNN_score_TSNE = kNN_acc_kfold(embedding_TSNE, L)
acc1, ari1, ami1 = kmeans_acc_ari_ami(embedding_org, L)
acc2, ari2, ami2 = kmeans_acc_ari_ami(embedding_hub, L)
acc3, ari3, ami3 = kmeans_acc_ari_ami(embedding_TSNE, L)
mantel_test(X, L, embedding_org)
mantel_test(X, L, embedding_hub)
mantel_test(X, L, embedding_TSNE)

print(kNN_score_org, kNN_score_hub, kNN_score_TSNE)
print([acc1, acc2, acc3], [ari1, ari2, ari3], [ami1, ami2, ami3])

visualize(embedding_org, L)
visualize(embedding_hub, L)
visualize(embedding_TSNE, L)

    # dataset = 'NORB'
    # n = X.shape[0]
    # # np.savez('embed_org_'+ dataset + str(n) + '_' + str(iter_n), X=X, L=L, emb=emb_org_list)
    # np.savez('embed_org_'+ dataset + str(n) + '_Seed:' + str(seed_lst[i]), X=X, L=L, emb=emb_org_list)
    # # np.savez('embed_hub_'+ dataset + str(n) + '_' + str(iter_n), X=X, L=L, emb=emb_hub_list)
    # np.savez('embed_hub_'+ dataset + str(n) + '_Seed:' + str(seed_lst[i]), X=X, L=L, emb=emb_hub_list)
# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embedding_org[:, 0], embedding_org[:, 1], c=color, cmap="viridis", s=0.1
# )
# plt.setp(ax, xticks=[], yticks=[])
# # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()
#
# fig, ax = plt.subplots(figsize=(12, 10))
# color = L.astype(int)
# plt.scatter(
#     embedding_hub[:, 0], embedding_hub[:, 1], c=color, cmap="viridis", s=0.1
# )
# plt.setp(ax, xticks=[], yticks=[])
# # plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
#
# plt.show()
# #

# # 1-NN
# X_train, X_test, Y_train, Y_test = train_test_split(embedding_hub, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# print("1-NN: ", score)
# # result_hub.append(score)
# #
# X_train, X_test, Y_train, Y_test = train_test_split(embedding_org, L, random_state=0)
# knc = KNeighborsClassifier(n_neighbors=1)
# knc.fit(X_train, Y_train)
# Y_pred = knc.predict(X_test)
# score = knc.score(X_test, Y_test)
# print("1-NN: ", score)
# # result_org.append(score)
# #
# # print(result_org, result_hub)
