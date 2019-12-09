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


sns.set(style='white', rc={'figure.figsize': (12, 10)})

# dataset = SmallNORBDataset(dataset_root='norb-small')

train_data = norb_loader.norb_data('train')
train_labels = norb_loader.norb_labels('train')
X = train_data
L = np.empty([train_labels.shape[0] * 2,], dtype=int)
L[0::2] = train_labels
L[1::2] = train_labels

print(X.shape, L.shape)
# #
# u = umap.UMAP(n_neighbors=30).fit_transform(X)
# plt.scatter(u[:, 0], u[:, 1], s=0.1, c=L, cmap='viridis')
# plt.show()

k = 30
seed = 42
result_hub = []
result_org = []

reducer = umap.UMAP(init='random', metric='precomputed', n_neighbors=k, random_state=seed)
embedding_hub = reducer.fit_transform(X)
# elapsed_time1 = time.time() - start1

# start2 = time.time()
reducer = umap.UMAP(n_neighbors=k, random_state=seed)
embedding_org = reducer.fit_transform(X)
# elapsed_time2 = time.time() - start2

fig, ax = plt.subplots(figsize=(12, 10))
color = L.astype(int)
plt.scatter(
    embedding_org[:, 0], embedding_org[:, 1], c=color, cmap="viridis", s=0.1
)
plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()

fig, ax = plt.subplots(figsize=(12, 10))
color = L.astype(int)
plt.scatter(
    embedding_hub[:, 0], embedding_hub[:, 1], c=color, cmap="viridis", s=0.1
)
plt.setp(ax, xticks=[], yticks=[])
# plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()

# 1-NN
X_train, X_test, Y_train, Y_test = train_test_split(embedding_hub, L, random_state=0)
knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, Y_train)
Y_pred = knc.predict(X_test)
score = knc.score(X_test, Y_test)
print("1-NN: ", score)
result_hub.append(score)

X_train, X_test, Y_train, Y_test = train_test_split(embedding_org, L, random_state=0)
knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, Y_train)
Y_pred = knc.predict(X_test)
score = knc.score(X_test, Y_test)
print("1-NN: ", score)
result_org.append(score)

print(result_org, result_hub)
