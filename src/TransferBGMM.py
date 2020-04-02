# Description: This program is designed to be helpful in cases where the 
# parameters of the microscope imaging have changed. This is designed to 
# allow a Bayesian Gaussian Mixture Model that was fit to a larger dataset
# with different imaging parameters to be used to classify images with these
# varied parameters. An example might be as follow:
#
# 10,000 images of a certain kind of TMD have been used to fit a BGMM.
# 200    images of a new kind of TMD are taken, but have a poor fit to
# the existing BGMM, because they occupy a very different part of the
# HSV space.
#
# The general idea is to find the transform that maximizes the mean log
# likelihood of the new dataset when scored against the old BGMM. This
# transform should be fairly simple. For this program, it will be a 
# translate, scale and rotate transform. The parameters of the transform
# will be
#     - translation vector:
#         (n_dimensions, 1)
#     - 


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
import torch

from mpl_toolkits.mplot3d  import Axes3D
from sklearn.decomposition import PCA
from sklearn.mixture       import BayesianGaussianMixture
from colors                import dist_colors             as colors



class TorchBGMM:
	def __init__(self, bgmm):
		self.prior_mean       = torch.tensor(bgmm.mean_prior).reshape(1, -1)
		self.means            = torch.tensor(bgmm.means_)
		self.prior_covariance = torch.tensor(bgmm.covariance_prior_)
		self.covariances      = torch.tensor(bgmm.covariances_)
		self.weights          = torch.tensor(bgmm.weights_)

	def predict_proba(self, X):
		X = X.T
		x_prior   = X - self.prior_mean.T
		prior_exp = (-x_prior * torch.matmul(self.prior_covariance, x_prior))
		prior_exp = prior_exp.sum(axis=0)
		prior     = torch.exp(prior_exp).reshape(-1, 1)

		# We need to repeat x along a third dimension so that we can
		# subtract the mean vector once for each component of the bgmm.
		x         = X.reshape((*X.shape, 1)).repeat(1, 1, 10)
		mean      = self.means.T.reshape((*self.means.T.shape, 1))
		mean      = mean.transpose(1, 2)
		x_shifted = x - mean

		

# Processes the arguments, sets up directories and validates that files
# actually exist.
def preprocess():
	parser = argparse.ArgumentParser(
		description='Test the accuracy of a PyTorch BGMM against it\'s ' +
		'corresponding counterpart.'
	)

	parser.add_argument(
		'-d', '--data', dest='data', type=str, required=True,
		help='The *.npy file of shape (n_samples, n_features) that will ' +
		'be used to test the performance of the model.'
	)

	parser.add_argument(
		'-m', '--model', dest='model', type=str, required=True,
		help='The pickled model file to load.'
	)

	args = parser.parse_args()

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

	# Construct a PCA reduction, if necessary.
	if data.shape[1] > 3:
		pca = PCA(n_components=3)
		pca.fit(data)

		data_reduced = pca.transform(data)
	else:
		data_reduced = data


	with open(args.model, 'rb') as file:
		bgmm = pickle.loads(file.read())




