from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# load and select relevant data
iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

# create scatterplot of sepal length vs. width by species
plt.scatter(X[:,0], X[:,1], c=Y)

# generate new random point
length = []
for el in enumerate(X):
	length.append(el[1][0])

width = []
for el in enumerate(X):
	width.append(el[1][1])

rand_point = []
rand_point.append(np.random.uniform(min(length), max(length)))
rand_point.append(np.random.uniform(min(width), max(width)))

# Calculate the euclidean distance from new point to all existing points
# sqrt((p1-q1)**2 + (p2-q2)**2)
distances = []
count = 0
for el in X:
	d = np.sqrt( (el[0]-rand_point[0])**2 + (el[1]-rand_point[1])**2 )
	distances.append([d, Y[count]])
	count = count + 1

# Sort distances and take top 10 which will be closest 10 points
distances.sort()
top10 = distances[:10]

# Determine the majority class of top 10
cls = []
for el in top10:
	cls.append(el[1])
Counter(cls)
# Majority class is 2

def prep_data():
	iris = datasets.load_iris()
	X = iris.data[:,:2]
	Y = iris.target

# ats = [random_sepal_length, random_sepal_width], k = # of neighbors
def knn(ats, k):
	prep_data()

	distances = []
	count = 0

	for el in X:
    		d = np.sqrt( (el[0]-ats[0])**2 + (el[1]-ats[1])**2 )
    		distances.append([d, Y[count]])
    		count = count + 1

	distances.sort()
	top = distances[:k]	

	cls = []
	for el in top:
    		cls.append(el[1])
	print(Counter(cls))
	print("0=serisota, 1=verisicolor, 2=virginica")
	print("Species: ", max(Counter(cls)))