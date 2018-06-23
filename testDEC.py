import numpy as np
from sklearn.cluster import KMeans
stacked_test = np.load('test_stacked.npy')
print(np.shape(stacked_test))
testNum = 20
maxWordNum = 410
featureS = np.reshape(stacked_test, (testNum,maxWordNum*10))
print(np.shape(featureS))
#===============
cls1 = KMeans(n_clusters=2)
fit = cls1.fit(featureS)
print(fit)
pred = cls1.fit_predict(featureS)
print(pred)
#==============
from xMeans import XMeans
x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten() #ランダムな80個の数を生成
y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten() #ランダムな80個の数を生成
xy = np.c_[x,y]
print(np.shape(xy))
x_means = XMeans(random_state = 1).fit(xy)  
print(x_means.labels_)
print(x_means.cluster_centers_)
print(x_means.cluster_log_likelihoods_)
print(x_means.cluster_sizes_)
#==============
x_means = XMeans().fit(featureS)  
print(x_means.labels_)
print(x_means.cluster_centers_)
print(x_means.cluster_log_likelihoods_)
print(x_means.cluster_sizes_)
