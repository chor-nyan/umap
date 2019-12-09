import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
# %matplotlib inline
sns.set(style='white', rc={'figure.figsize':(12,8)})

import requests
import zipfile
import imageio
import os
import re
from skhubness.neighbors import kneighbors_graph
import hub_toolbox
from hub_toolbox.distances import euclidean_distance

import umap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import MulticoreTSNE
import sklearn.manifold
import random
from numpy import savetxt

if not os.path.isfile('coil20.zip'):
    results = requests.get('http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip')
    with open("coil20.zip", "wb") as code:
        code.write(results.content)


images_zip = zipfile.ZipFile('coil20.zip')
mylist = images_zip.namelist()
r = re.compile(".*\.png$")
filelist = list(filter(r.match, mylist))

# if not os.path.isfile('coil-20-proc/obj10__1.png'):
#     unzip coil20.zip

feature_vectors = []
for filename in filelist:
    im = imageio.imread(filename)
    feature_vectors.append(im.flatten())

labels = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
labels = labels.astype(int)
data = pd.DataFrame(feature_vectors, index=labels)
# data.shape
k = 5
# seed = 0
data = data.values
data = data / 255.

result_hub = []
result_org = []

iter_n = 5
seed_lst = random.sample(range(100), k=iter_n)
for i in range(iter_n):
    # seed = random.randint(0, 100)
    seed = seed_lst[i]

    fit = umap.UMAP(init='random', metric='euclidean', n_neighbors=k, n_epochs=2000, random_state=seed, min_dist=0.5)
    u_org = fit.fit_transform(data)

    D = euclidean_distance(data)
    fit = umap.UMAP(init='random', n_neighbors=k, metric='precomputed', n_epochs=2000, random_state=seed, min_dist=0.5)
    u_hub = fit.fit_transform(D)
# neigbour_graph = kneighbors_graph(data, algorithm='hnsw', algorithm_params={'n_candidates': 100}, n_neighbors=k,
#                                   mode='distance', hubness='mutual_proximity',
#                                   hubness_params={'method': 'normal'})
# u = fit.fit_transform(D_mp)

    plt.scatter(u_org[:,0], u_org[:,1], c=labels, cmap="Spectral", s=10)
    plt.show()

    plt.scatter(u_hub[:,0], u_hub[:,1], c=labels, cmap="Spectral", s=10)
    plt.show()

    # 1-NN
    X_train, X_test, Y_train, Y_test = train_test_split(u_org, labels, random_state=0)
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, Y_train)
    Y_pred = knc.predict(X_test)
    score = knc.score(X_test, Y_test)
    # print(score)
    result_org.append(score)

    # 1-NN
    X_train, X_test, Y_train, Y_test = train_test_split(u_hub, labels, random_state=0)
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(X_train, Y_train)
    Y_pred = knc.predict(X_test)
    score = knc.score(X_test, Y_test)
    # print(score)
    result_hub.append(score)
    print(result_org)
    print(result_hub)
savetxt('/home/hino/git/umap2/examples/result_org_COIL20.csv', result_org)
savetxt('/home/hino/git/umap2/examples/result_hubE_COIL20.csv', result_hub)
