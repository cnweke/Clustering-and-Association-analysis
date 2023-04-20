import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.datasets import load_iris

iris = load_iris()
iris.data
iris.target

#We specify three clusters
kmeans = KMeans(n_clusters=3)
kmeans

 # we'll fit a model using training data
KModel = kmeans.fit(iris.data)
KModel2 = kmeans.fit_predict(iris.data) #Kmeans clustering output
print(KModel)

KModel.labels_
# the center of clusters
KModel.cluster_centers_

pd.crosstab(iris.target,KModel.labels_)

#Find distance between points of cluster

kmeans.inertia_

df = pd.DataFrame(iris.data)
df['cluster'] = KModel2
df['cluster'].value_counts()

plt.scatter(iris.data[:,0],iris.data[:,1], c = KModel2, cmap='rainbow')
plt.show()