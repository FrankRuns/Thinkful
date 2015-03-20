from sklearn import datasets, svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import ipdb

# load data
iris = datasets.load_iris()

# chose flower features for classification
feat1 = 0
feat2 = 2

# graph feat1 against feat2
plt.scatter(iris.data[:,feat1], iris.data[:,feat2], c=iris.target[:])
plt.xlabel(iris.feature_names[feat1])
plt.ylabel(iris.feature_names[feat2])
plt.show()

# fit the svm on feat1 and feat2 data
svc = svm.SVC(kernel='linear')
X = iris.data[:, [feat1,feat2]]
y = iris.target[:]
svc.fit(X, y)

# visualize the svm results
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
	estimator.fit(X,y)
	x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
	y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
		                 np.linspace(y_min, y_max, 100))
	Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	plt.scatter(X[:,0], X[:,1], c=iris.target[:], cmap=cmap_bold)
	plt.axis('tight')
	#plt.axis('off')
	plt.tight_layout()
	plt.xlabel(iris.feature_names[1])
	plt.ylabel(iris.feature_names[2])
	plt.show()

plot_estimator(svc, X, y)

