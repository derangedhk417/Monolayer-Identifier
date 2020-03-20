# Description: This program constructs a sample dataset suitable for
# testing against the TrainBGMM.py script. It allows you to specify the
# dimensionality of the dataset, the number of clusters and the minimum
# distance (Euclidean) between clusters.
# 
# Example Output:
# array([[-0.08189495,  1.19018641, -0.63447915],
#        [ 1.09406794,  0.4725111 ,  0.67889614],
#        [ 0.9118422 , -0.49732009, -1.31487465],
#        [ 0.95954403, -1.30263586, -0.25844434],
#        [ 0.85958051,  1.19474884,  0.30796178],
#        [-0.36416368, -1.05787487,  1.00366701],
#        [-0.96999955,  0.64697356, -1.05135048],
#        [-1.73889003,  1.35270568, -0.96519018]])
#
# This example is just a single cluster centered at (0, 0, 0) with
# sigma = 1.0
#
# Notes:
#     - This program will also output the parameters of each cluster in
#       two files. The first will be a human-readable *.csv file. The second
#       will be a machine readable *.npy file. The names will be
#       output.meta.csv and output.met.npy.
#     - The *.npy file is useful for regenerating samples from the same
#       distribution. It is also useful for slightly modifying the parameters
#       of the distributions, generating a new dataset and then assessing how
#       well a transfer learning method establishes a correlation between 
#       clusters in the modified dataset and clusters in the original dataset.

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
from colors                import dist_colors  as colors
from readable_csv          import readable_csv as csv

# Processes the arguments, sets up directories and validates that files
# actually exist.
def preprocess():
	parser = argparse.ArgumentParser(
		description='Construct a sample dataset for testing with TrainBGMM.py'
	)

	parser.add_argument(
		'-o', '--output', dest='output', type=str, required=True,
		help='The output file to write the dataset to.'
	)

	parser.add_argument(
		'-n', '--n-dims', dest='n_dimensions', type=int, required=True,
		help='The number of dimensions of the dataset.'
	)

	parser.add_argument(
		'-c', '--n-clusters', dest='n_clusters', type=int, required=True,
		help='The number of clusters to generate.'
	)

	parser.add_argument(
		'-m', '--mean-cluster-members', dest='mean_cluster_members', type=int,
		default=256, help='The mean number of members to give to each cluster.'
	)

	parser.add_argument(
		'-s', '--std-cluster-members', dest='std_cluster_members', type=int,
		default=128, help='The standard deviation in the number of members ' +
		'per cluster.'
	)

	parser.add_argument(
		'-d', '--minimum-distance', dest='minimum_distance', type=float,
		default=2.0, help='The minimum Euclidean distance between the ' +
		'mean location of any two clusters.'
	)

	parser.add_argument(
		'-g', '--std-location', dest='std_location', type=float,
		default=0.0, help='The standard deviation in the location of the ' +
		'mean of each cluster. If this isn\'t set by the user, it will be ' +
		'set to n_clusters*(minimum_distance^(1 / n_dimensions))'
	)

	parser.add_argument(
		'-sm', '--std-mean', dest='mean_of_std', type=float,
		default=0.0, help='The mean value of the elements of the covariance ' +
		'matrix for each distribution that each cluster is sampled from. ' +
		'This will default to 0.8 * minimum_distance if it isn\'t set.'
	)

	parser.add_argument(
		'-ss', '--std-std', dest='std_of_std', type=float,
		default=0.0, help='The standard deviation of the standard deviation ' +
		'of the clusters. This is essentialy the variation in the elements ' +
		'of the covariance matrix for each cluster\'s distribution. Default ' +
		'is half of the mean of the standard deviation (-sm/--std-mean)'
	)

	args = parser.parse_args()

	if args.std_location == 0.0:
		args.std_location = args.n_clusters * np.power(
			args.minimum_distance, 1 / args.n_dimensions
		)

	if args.mean_of_std == 0.0:
		args.mean_of_std = 0.8 * args.minimum_distance

	if args.std_of_std == 0.0:
		args.std_of_std = 0.5 * args.mean_of_std

	# Validate the output file.
	if os.path.exists(args.output):
		f = args.output
		print("The specified output file \'%s\' already exists"%f)
		print("Either delete it, or set the output manually with -o/--output")
		exit()

	return args

if __name__ == '__main__':
	args = preprocess()

	# Start by generating a center location for each cluster.
	center_locations = np.ones((args.n_clusters, args.n_dimensions))*1e6

	# Generate center locations in a loop, checking that each one meets the
	# minimum distance constraint and retrying if it doesn't
	n_complete = 0
	while n_complete < args.n_clusters:
		center_locations[n_complete, :] = np.random.normal(
			0.0,
			args.std_location,
			(args.n_dimensions,)
		)

		# Calculate the smallest distance between this center and every other
		# center.
		distances = (center_locations[n_complete, :] - center_locations)
		distances = np.sqrt((distances**2).sum(axis=1))

		# Filter out the zero distance, because that is just the center compared
		# to itself.
		distances    = distances[distances > 1e-9]
		min_distance = distances.min()

		# Only move to the next center location if this one meets criteria.
		if min_distance > args.minimum_distance:
			n_complete += 1


	# Now that we have a set of center locations, we generate a standard 
	# deviation along each dimension, for each cluster.
	standard_deviations = np.abs(np.random.normal(
		args.mean_of_std, 
		args.std_of_std,
		(args.n_clusters, args.n_dimensions)
	))

	# Now that we have parameters for each cluster, its time to actually sample
	# each distribution and produce those clusters.

	# Here we store all of the data in one large array that can be written to 
	# the disk. The second array stores each cluster separately so that they
	# can each be plotted in a multicolored scatter plot.
	data     = np.zeros((0, args.n_dimensions))
	clusters = []

	# These are used to store the meta information that gets dumped to
	# another *.npy file and to a csv file.
	sizes    = np.zeros((args.n_clusters, 1))
	indices  = np.zeros((args.n_clusters, 1))

	for cluster_idx in range(args.n_clusters):
		n_samples       = int(abs(np.random.normal(
			args.mean_cluster_members,
			args.std_cluster_members
		)))

		sizes[cluster_idx, 0]   = float(n_samples)
		indices[cluster_idx, 0] = float(cluster_idx) 

		current_cluster = np.random.normal(
			center_locations[cluster_idx],
			standard_deviations[cluster_idx],
			(n_samples, args.n_dimensions)
		)

		clusters.append(current_cluster)
		data = np.concatenate((data, current_cluster))

	# Now that we have generated the data, apply principle component
	# analysis to reduce the dimensionality (if necessary) and then
	# produce a 3D scatterplot of it.

	if args.n_dimensions > 3:
		pca = PCA(n_components=3)
		pca.fit(data)
		data_reduced = [pca.transform(cluster) for cluster in clusters]
	else:
		data_reduced = clusters

	fig = plt.figure()
	ax  = fig.add_subplot(111, projection='3d')

	for idx, cluster in enumerate(clusters):
		color = colors[idx]

		ax.scatter(
			cluster[:, 0],
			cluster[:, 1],
			cluster[:, 2],
			s=3, c=color
		)

	ax.set_title("Cluster Visualization")
	plt.show()

	# Now we save the *.npy file and also create a *.stats file where
	# we dump a csv file with the mean, covariance matrix diagonal and
	# number of members per cluster.

	np.save(args.output, data)

	column_names = ["#"]
	column_names.append("N")
	column_names.extend(["mu_%d"%i for i in range(args.n_dimensions)])
	column_names.extend(["sigma_%d"%i for i in range(args.n_dimensions)])

	meta_data = np.concatenate((
		indices, 
		sizes, 
		center_locations, 
		standard_deviations
	), axis=1)

	np.save(args.output + '.meta.npy', meta_data)

	with open(args.output + '.meta.csv', 'w') as file:
		file.write(csv(meta_data.tolist(), column_names, digits=3))

