# Description: This program is designed to take a directory containing
# one or more images and convert it into a *.npy file and corresponding
# *.csv file, both of which contain the location of every "flake" in 
# every image in the folder. 

import code
import sys
import argparse
import os
import time
import cv2
import pickle
import numpy             as np
import matplotlib.pyplot as plt
from sklearn.mixture       import BayesianGaussianMixture
from readable_csv          import readable_csv as csv

def torgb(h):
	return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]

# Processes the arguments, sets up directories and validates that files
# actually exist.
def preprocess():
	parser = argparse.ArgumentParser(
		description='Extract contour information from a folder of images.'
	)

	parser.add_argument(
		'-o', '--output', dest='output', type=str, required=True,
		help='The output file to write the data to.'
	)

	parser.add_argument(
		'-d', '--directory', dest='directory', type=str, required=True,
		help='The directory to read images from.'
	)

	parser.add_argument(
		'-s', '--show', dest='show_image', type=str, default='',
		help='Display a plot of contours extracted from this image.'
	)

	args = parser.parse_args()

	return args


def draw_contours(image, contours, edges):
	cpy = image.copy()
	for i in range(len(contours)):
		cpy = cv2.drawContours(cpy, contours, i, (255, 0, 0), 1)

	plt.subplot(131)
	plt.imshow(image)
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(132)
	plt.imshow(edges)
	plt.title('Edge Detection')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(133)
	plt.imshow(cpy)
	plt.title('Contours')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	plt.show()

def getLargerThan(contours, s):
	selected = []

	for c in contours:
		area = cv2.contourArea(c)
		if area > s:
			selected.append(c)

	return selected

# TODO: Make the Canny edge detect parameters adjustable.
# TODO: Make the contour method adjustable.
def extract_data_from_file(fname):
	img   = cv2.imread(fname, cv2.IMREAD_COLOR)

	#code.interact(local=locals())

	# Try to reduce the image size to around 300 x 300
	scale = np.sqrt((300**2) / (img.shape[0] * img.shape[1]))

	img   = cv2.resize(img, (0, 0), fx = scale, fy = scale)
	plt.subplot(231)
	plt.imshow(img)
	plt.title("Original Image (Downscaled)")
	plt.xticks([])
	plt.yticks([])

	img   = cv2.GaussianBlur(img, (5, 5), 2)
	plt.subplot(232)
	plt.imshow(img)
	plt.title("Original Image (Gaussian Blurred)")
	plt.xticks([])
	plt.yticks([])

	

	# Reshape the image into something that can be processed by the clustering
	# algorithm.

	data = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

	bgmm = BayesianGaussianMixture(
		n_components=3,
		covariance_type='full'
	)

	bgmm.fit(data)
	classes = bgmm.predict(data)
	# Convert the three classes into regions that are red, green or blue.
	new_colors = img.copy()
	new_colors[:, :, 0] = 255 * (classes == 0).reshape(img.shape[0], img.shape[1])
	new_colors[:, :, 1] = 255 * (classes == 1).reshape(img.shape[0], img.shape[1])
	new_colors[:, :, 2] = 255 * (classes == 2).reshape(img.shape[0], img.shape[1])


	#code.interact(local=locals())

	grad  = cv2.Sobel(img, -1, 1, 1, ksize=1)

	plot_grad = grad.copy()
	plot_grad = (plot_grad - plot_grad.min()) / (plot_grad.max() - plot_grad.min())
	plt.subplot(233)
	plt.imshow(plot_grad)
	plt.title("Sobel Filter")
	plt.xticks([])
	plt.yticks([])


	lower = grad.max() * 0.45
	upper = grad.max() * 0.75

	print(lower)
	print(upper)


	edges = cv2.Canny(img, lower, upper)

	plt.subplot(234)
	plt.imshow(edges)
	plt.title("Edge Detected Image")
	plt.xticks([])
	plt.yticks([])

	# contours, heirarchy = cv2.findContours(
	# 	edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
	# )
	

	contours, heirarchy = cv2.findContours(
		edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)

	#contours = getLargerThan(contours, 80.0)

	
	# draw_contours(img, contours, edges)

	cpy = img.copy()
	for i in range(len(contours)):
		cpy = cv2.drawContours(cpy, contours, i, (255, 0, 0), 1)

	plt.subplot(235)
	plt.imshow(cpy)
	plt.title('Contours')
	plt.xticks([])
	plt.yticks([])

	plt.subplot(236)
	plt.imshow(new_colors)
	plt.title('Clustered')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout(rect=[0, 0, 0, 0])
	plt.show()


if __name__ == '__main__':
	args = preprocess()

	# List of pairs. First element is the index of the
	# contour in the exported *.npy file. Second element
	# is the name of the file that it came from.
	contour_indices = []

	# This stores the following for every contour found.
	#     1) X    - coordinate of center
	#     2) Y    - coordinate of center
	#     3) Area - in pixels
	#     4) Hue
	#     5) Saturation
	#     6) Value

	results = []

	# The files to read.
	files = [f for f in os.listdir(args.directory)]

	# Filter to only actual files.
	tmp = []
	for f in files:
		if os.path.isfile(os.path.join(args.directory, f)):
			tmp.append(f)
	files = sorted(tmp)

	for file in files:
		extract_data_from_file(os.path.join(args.directory, file))



















