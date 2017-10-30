import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np 


def update_data(X,X1):
	return np.vstack((X,X1))

class K_means:
	def __init__(self,k=10,tol=.001,max_iter=10):
		self.k=k
		self.tol=tol
		self.max_iter=max_iter


	def fit(self,data):
		self.centroids={}

		for i in range(self.k):
			self.centroids[i]=data[i]

		for i in range(self.max_iter):
			self.classifications={}

			for i in range(self.k):
				self.classifications[i]=[]
			
			for featureset in data:
				
				distances=[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids=dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized=True

			for c in self.centroids:
				original_centroid=prev_centroids[c]
				current_centroid=self.centroids[c]

				if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
					optimized = False

			if optimized:
				break

	def predict(self,data):

		distances=[np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification
		

if __name__ == '__main__':

	sample_size=100
	amplifier=100

	X=np.random.random((sample_size,2))*amplifier

	number_of_classes=6

	clf=K_means(k=number_of_classes)
	clf.fit(X)

	classes_color_number=clf.k
	colors=classes_color_number*["g","r","b","y","c","b","k"]

	

	for el in clf.classifications:
		plt.scatter(clf.centroids[el][0],clf.centroids[el][1],color=colors[el])
		for element in clf.classifications[el]:
			plt.scatter(element[0],element[1],color=colors[el])
	
	plt.show()







