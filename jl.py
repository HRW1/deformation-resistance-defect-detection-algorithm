import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('E:/Clion/def_det/30.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
r,c = gray.shape

indx, indy = np.where(gray == 255)
indx = indx.reshape(-1,1)
indy = indy.reshape(-1,1)
X = np.hstack((indy,r-indx))

A = [[c-1,r-1],[0,r-1],[c-1,0],[0,0]]
A = np.asarray(A)
print(A.shape)


# 计算DBSCAN
db = DBSCAN(eps=20, min_samples=4).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# 聚类的结果
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of samples: %d' %len(X))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# 绘出结果
unique_labels = set(labels)
color=['red','green','blue']
shape=['^','o','s']

plt.scatter(A[:, 0], A[:, 1], s=1, c='#000000')

for k in unique_labels:
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    num,_ = xy.shape
    # if num < 5 :
    #     continue
    # print('%d,%d', k, num)
    plt.scatter(xy[:, 0], xy[:, 1], s=30, c='none', marker=shape[k%len(shape)],edgecolors=color[k%len(color)],linewidths=0.5)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=30, c='black', marker='x',linewidths=0.5)
plt.show()