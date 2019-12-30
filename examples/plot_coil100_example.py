import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
# %matplotlib inline
# sns.set(style='white', rc={'figure.figsize':(12,8)})
sns.set()
sns.set_style('whitegrid')
sns.set_palette('gray')

import requests
import zipfile
import imageio
import os
import re
from skhubness.neighbors import kneighbors_graph
import hub_toolbox
from hub_toolbox.distances import euclidean_distance
# from utils import global_score, mantel_test

import umap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import MulticoreTSNE
import sklearn.manifold
import random
from numpy import savetxt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style='white', rc={'figure.figsize':(12,8)})

import requests
import tarfile
import imageio
import cv2
import glob
import os
import time

import umap

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from evaluate import kNN_acc, kNN_acc_kfold, kmeans_acc_ari_ami, visualize, mantel_test
from openTSNE import TSNE

# if not os.path.isfile('coil20.zip'):
#     results = requests.get('http://www.cs.columbia.edu/CAVE/Xbases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip')
#     with open("coil20.zip", "wb") as code:
#         code.write(results.content)

if not os.path.exists('coil-100'):
    results = requests.get('http://www.cs.columbia.edu/CAVE/Xbases/SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz')
    with open("coil_100.tar.gz", "wb") as code:
        code.write(results.content)

    images_zip = tarfile.open('coil_100.tar.gz', mode='r:gz')
    images_zip.extractall()

feature_vectors = []
filelist = glob.glob('./coil-100/*.ppm')
for filename in filelist:
    im = cv2.imread(filename)
    feature_vectors.append(im.flatten())

L = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
L = L.to_numpy(dtype='int32')
# print(L.shape, L.dtype)
X = np.vstack(feature_vectors)
X = X / 255.

# print(X.shape)

# images_zip = zipfile.ZipFile('coil20.zip')
# mylist = images_zip.namelist()
# r = re.compile(".*\.png$")
# filelist = list(filter(r.match, mylist))

# if not os.path.isfile('coil-20-proc/obj10__1.png'):
#     unzip coil20.zip

# feature_vectors = []
# for filename in filelist:
#     im = imageio.imread(filename)
#     feature_vectors.append(im.flatten())
#
# L = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
# L = L.astype(int)
# X = pd.DataFrame(feature_vectors, index=L)
# X.shape
k = 5
# seed = 0
# X = X.values

# result_hub = []
# result_org = []

emb_org_list = []
emb_hub_list = []

pca = False
if pca:
    X -= np.mean(X, axis=0)
    X = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)

print("X shape:", X.shape)

iter_n = 1

seed_lst = random.sample(range(100), k=iter_n)
seed_set = set(seed_lst)
# seed_set.discard(97)
# seed_set.discard(69)
# seed_set.discard(99)
seed_lst = list(seed_set)
print(seed_lst)
for i in range(iter_n):
    # seed = random.randint(1, 100)
    seed = seed_lst[i]

    # start2 = time.time()
    # # D = euclidean_distance(X)
    # fit = umap.UMAP(init='random', n_neighbors=k, metric='precomputed', n_epochs=1000, random_state=seed, min_dist=0.5)
    # # fit = umap.UMAP(n_neighbors=k, metric='precomputed', n_epochs=1000, random_state=seed, min_dist=0.5)
    # # embedding_hub = fit.fit_transform(D)
    # embedding_hub = fit.fit_transform(X)
    # elapsed_time2 = time.time() - start2
    #
    #
    # start1 = time.time()
    # fit = umap.UMAP(init='random', metric='euclidean', n_neighbors=k, n_epochs=1000, random_state=seed, min_dist=0.5)
    # embedding_org = fit.fit_transform(X)
    # elapsed_time1 = time.time() - start1

    start3 = time.time()
    embedding_TSNE = TSNE().fit(X)
    elapsed_time3 = time.time() - start3

    # emb_org_list.append(embedding_org)
    # emb_hub_list.append(embedding_hub)
    #
    # print('org', elapsed_time1)
    # print('hub', elapsed_time2)
    print('TSNE:', elapsed_time3)

    # np.savez('embed_org_' + 'coil100' + '_' + str(iter_n), X=X, L=L, emb=emb_org_list)
    # np.savez('embed_hub_' + 'coil100' + '_' + str(iter_n), X=X, L=L, emb=emb_hub_list)

    # np.savez('embed_org_' + 'coil100' + '_Seed:' + str(seed), X=X, L=L, emb=embedding_org)
    # np.savez('embed_hub_' + 'coil100' + '_Seed:' + str(seed), X=X, L=L, emb=embedding_hub)

# kNN_score_org = kNN_acc_kfold(embedding_org, L)
# kNN_score_hub = kNN_acc_kfold(embedding_hub, L)
kNN_score_TSNE = kNN_acc_kfold(embedding_TSNE, L)

# acc1, ari1, ami1 = kmeans_acc_ari_ami(embedding_org, L)
# acc2, ari2, ami2 = kmeans_acc_ari_ami(embedding_hub, L)
acc3, ari3, ami3 = kmeans_acc_ari_ami(embedding_TSNE, L)

# print(kNN_score_org, kNN_score_hub, kNN_score_TSNE)
# print([acc1, acc2, acc3], [ari1, ari2, ari3], [ami1, ami2, ami3])

print(kNN_score_TSNE, acc3, ari3, ami3)

# mantel_test(X, L, embedding_org)
# mantel_test(X, L, embedding_hub)
mantel_test(X, L, embedding_TSNE)

# visualize(embedding_org, L)
# visualize(embedding_hub, L)
visualize(embedding_TSNE, L)
# neigbour_graph = kneighbors_graph(X, algorithm='hnsw', algorithm_params={'n_candidates': 100}, n_neighbors=k,
#                                   mode='distance', hubness='mutual_proximity',
#                                   hubness_params={'method': 'normal'})
# u = fit.fit_transform(D_mp)

#     plt.scatter(embedding_org[:,0], embedding_org[:,1], c=L, cmap="Spectral", s=10, alpha=0.5)
#     plt.show()
#
#     plt.scatter(embedding_hub[:,0], embedding_hub[:,1], c=L, cmap="Spectral", s=10, alpha=0.5)
#     plt.show()
#
#     # 1-NN
#     X_train, X_test, Y_train, Y_test = train_test_split(embedding_org, L, random_state=0)
#     knc = KNeighborsClassifier(n_neighbors=1)
#     knc.fit(X_train, Y_train)
#     Y_pred = knc.predict(X_test)
#     score = knc.score(X_test, Y_test)
#     # print(score)
#     result_org.append(score)
#
#     # 1-NN
#     X_train, X_test, Y_train, Y_test = train_test_split(embedding_hub, L, random_state=0)
#     knc = KNeighborsClassifier(n_neighbors=1)
#     knc.fit(X_train, Y_train)
#     Y_pred = knc.predict(X_test)
#     score = knc.score(X_test, Y_test)
#     # print(score)
#     result_hub.append(score)
#
#     print(result_org)
#     print(result_hub)
#
# savetxt('/home/hino/git/umap2/examples/result_org_COIL100.csv', result_org)
# savetxt('/home/hino/git/umap2/examples/result_hub_COIL100.csv', result_hub)

    # r_lst_org, p_org = mantel_test(X, L, embedding_org)
    # r_lst_hub, p_hub = mantel_test(X, L, embedding_hub)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.boxplot([r_lst_org, r_lst_hub], L=['original', 'with HR'])
    # # ax.set_xlabel('methods')
    # ax.set_ylabel('PCC')
    # ax.set_ylim(0, 1)
    #
    # plt.show()
    #
    # print(p_org)
    # print(p_hub)


# time:
# org 68.78906083106995
# hub 329.6923770904541
# org 67.5874011516571
# hub 400.18688797950745
# org 65.88189697265625
# hub 403.05447793006897
# org 67.63822793960571
# hub 407.7483446598053
# org 65.31972098350525
# hub 362.7240858078003
# org 64.45434308052063
# hub 359.3625190258026
# org 65.47392177581787
# hub 352.7872951030731
# org 65.82275700569153
# hub 348.25289511680603
# org 65.4333028793335
# hub 349.73782086372375
# org 64.4721052646637
# hub 350.36957240104675