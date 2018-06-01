"""
	unsupervised learning algorithm
	-	random initialize centroids
"""
from sklearn.cluster import MiniBatchKMeans

class KMean(object):
	def __init__(self, n_clusters = 3):
		self.cluster = MiniBatchKMeans(n_clusters = n_clusters, random_state = 0)

	def fit(self, X):
		self.cluster.partial_fit(X)

	def transform(self, x):
		return self.cluster.predict(x)

	def predict(self, *args, **kwargs):
		pass

	def save(self, *args, **kwargs):
		pass