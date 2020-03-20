# Description: This program is designed to take a dataset as an argument
# and training a Bayesian-Gaussian Mixture Model. The dataset given to this
# program should be a *.npy file containing a 2 Dimensional array. The shape
# of the array should be (n_samples, n_features) where
#     n_samples  = The number of data points to use in the BGMM
#     n_features = The number of dimensions for each data point
# 
# Example:
#     - This example is for a set of Hue, Saturation and Value data points
#       for 8 crystals identified by a computer vision algorithm. This is the
#       most likely use case for this code.
#     - In actuality, hundreds if not thousands of samples are necessary to 
#       train a BGMM.
#     
# array([[214.77163177,  17.708749  ,  36.47292799],
#        [ 79.90910834,  88.53088496,  71.65092714],
#        [257.10427665,  19.55513927,  82.85824268],
#        [ 86.11374554,  26.96281794,  36.44603096],
#        [ 59.5689323 ,  67.97427278,  66.95306468],
#        [233.79161404,  71.65866931,  57.6562444 ],
#        [ 74.75576139,  73.53261482,  61.35433548],
#        [312.05402261,  56.62499227,  21.57079712]])
#
# Results of this code:
#     - This code will produce a BGMM model, serialized into a *.pickle
#       file for later use. The location where this file is stored will
#       default to "bgmm.pickle". You can set the location where this
#       file goes using the -o/--output argument.
#     - This file will also produce and display a 3D scatterplot of the
#       resulting clusters. Each cluster will have a different color. 
#       If n_features is greater than 3, Principle Component Analysis will
#       be used to reduce the dimensionality of the final data and display 
#       it in three dimensions.
#
# Notes:
#     - If you want to visualize your data in 3 Dimensions in order to decide 
#       on how many clusters you want the code to produce, you can do so by 
#       running the program as follows:
#
#           python3 TrainBGMM.py -i test.npy -v

import warnings
warnings.filterwarnings("ignore")

import code
import sys
import argparse
import os
import time
import pickle

import numpy             as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d  import Axes3D
from sklearn.decomposition import PCA
from sklearn.mixture       import BayesianGaussianMixture
from colors                import dist_colors             as colors

# Processes the arguments, sets up directories and validates that files
# actually exist.
def preprocess():
	parser = argparse.ArgumentParser(
		description='Construct and display a Bayesian Gaussian Mixture ' +
		'model from the given dataset.'
	)

	parser.add_argument(
		'-i', '--input', dest='training_data', type=str, required=True,
		help='The *.npy file of shape (n_samples, n_features) that will ' +
		'be used to construct the model.'
	)

	parser.add_argument(
		'-o', '--output', dest='output_file', type=str, default='bgmm.pickle',
		help='The file to write the final model to. This program will not ' +
		'overwrite an existing file. Instead, it will terminate if the ' +
		'output file already exists.'
	)

	parser.add_argument(
		'-m', '--mixture-components', dest='n_mixture_components', type=int,
		default=10, help='The maximum number of components (cluster) in the ' +
		'Bayesian Gaussian Mixture Model that will be constructed.'
	)

	parser.add_argument(
		'-v', '--visualize', dest='visualize', action='store_true',
		help='Don\'t construct a model, just plot the data'
	)

	args = parser.parse_args()

	# Validate the output file.
	if os.path.exists(args.output_file):
		f = args.output_file
		print("The specified output file \'%s\' already exists"%f)
		print("Either delete it, or set the output manually with -o/--output")
		exit()

	return args

if __name__ == '__main__':
	# Start by processing arguments.
	args = preprocess()

	# Load the data.
	try:
		data = np.load(args.training_data)
	except Exception as ex:
		f = args.training_data
		print("The specified input file \'%s\' could not be loaded."%f)
		print("Exception: %s"%str(ex))
		exit()

	# Check the data.
	if len(data.shape) != 2:
		print("This program requires a dataset of shape ", end='')
		print("(n_samples, n_features).")
		print("The specified file has %d dimensions"%(len(data.shape)), end='')
		print(" but this program expects 2.")
		exit()
	if data.shape[1] > 3:
		print("NOTICE: The dataset has more than 3 features. Principle ")
		print("Component Analysis will be used to reduce dimensionality ")
		print("before plotting any data.")
	if data.shape[1] < 3:
		print("This program requires that n_features be at least 3.")
		print("The specified dataset has only %d features."%(data.shape[1]))
		exit()


	# Construct a PCA reduction, if necessary.
	if data.shape[1] > 3:
		pca = PCA(n_components=3)
		pca.fit(data)

		data_reduced = pca.transform(data)
	else:
		data_reduced = data


	if args.visualize:
		# Just make a 3d scatterplot of the data.
		fig = plt.figure()
		ax  = fig.add_subplot(111, projection='3d')

		ax.scatter(
			data_reduced[:, 0],
			data_reduced[:, 1],
			data_reduced[:, 2],
			s=3
		)

		ax.set_title("3D Dataset Visualization")
		plt.show()

	else:
		# Construct a BGMM and then plot it.
		bgmm = BayesianGaussianMixture(
			n_components=args.n_mixture_components,
			covariance_type='full'
		)

		bgmm.fit(data)

		if not bgmm.converged_:
			print("NOTE: The fit procedure did not converge. This usually")
			print("doesn\'t matter.")

		fig = plt.figure()
		ax  = fig.add_subplot(111, projection='3d')

		classes = bgmm.predict(data)

		# Count the number of members of each cluster and inform
		# the user of how many clusters were actually identified.
		counts = np.zeros(args.n_mixture_components)
		for cluster_idx in range(args.n_mixture_components):
			count = (classes == cluster_idx).sum()
			counts[cluster_idx] = count

		# Consider the cluster significant if more than 1% of the data
		# resides in it.
		n_found = (counts > int(0.01 * data.shape[0])).sum()

		print("Fit procedure found %d of %d specified clusters."%(
			n_found, args.n_mixture_components
		))

		for cluster_idx in range(args.n_mixture_components):
			color   = colors[cluster_idx]
			mask    = classes == cluster_idx
			to_plot = data_reduced[mask]

			ax.scatter(
				to_plot[:, 0],
				to_plot[:, 1],
				to_plot[:, 2],
				s=3, c=color
			)


		ax.set_title("3D Cluster Visualization")
		plt.show()

		# Write the model to a file.
		with open(args.output_file, 'wb') as file:
			file.write(pickle.dumps(bgmm))


