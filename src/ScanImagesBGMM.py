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
from colors import dist_colors as colors

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

def plotimg(img, position, name):
	plt.subplot(position)
	plt.imshow(img)
	plt.title(name)
	plt.xticks([])
	plt.yticks([])

def subimage(image, rect):
	border_size = int(image.shape[0] / 2)
	image = cv2.copyMakeBorder(
		image,
		border_size,
		border_size,
		border_size,
		border_size,
		cv2.BORDER_CONSTANT,
		value=[0, 0, 0]
	)

	((x, y), (w, h), theta) = rect

	x = x + border_size
	y = y + border_size

	size = (image.shape[1], image.shape[0])

	rotation_matrix = cv2.getRotationMatrix2D(center=(x, y), angle=theta, scale=1)
	new_image       = cv2.warpAffine(image, rotation_matrix, dsize=size)

	x = int(x - w/2)
	y = int(y - h/2)

	w = int(w)
	h = int(h)

	result = new_image[y:y+h, x:x+w, :]
	#code.interact(local=locals())
	return result


def extract_flakes(fname):
	n_classes          = 2
	denoise_strength   = 15
	edge_dilate_size   = 3
	mask_erode_size    = 12
	contour_thickness  = 3
	filter_edge_length = 0.02
	border_width       = 5

	img = cv2.imread(fname, cv2.IMREAD_COLOR)
	# img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

	# Make a downscaled image to run the BGMM training on.
	bgmm_scale = np.sqrt((100**2) / (img.shape[0] * img.shape[1]))
	img_mini   = cv2.resize(img, (0, 0), fx=bgmm_scale, fy=bgmm_scale)


	# Make a downscaled image to do the segmentation on. This one won't 
	# be as small.
	scale     = np.sqrt((1000**2) / (img.shape[0] * img.shape[1]))
	img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)

	# Denoise the scaled down image before training the BGMM.
	img_mini = cv2.fastNlMeansDenoisingColored(
		img_mini, 
		denoise_strength, 
		denoise_strength
	)

	# Train a BGMM on the really small image.
	samples = img_mini.reshape(
		img_mini.shape[0] * img_mini.shape[1], 
		img_mini.shape[2]
	)

	bgmm = BayesianGaussianMixture(
		n_components=n_classes,
		covariance_type='full'
	)

	bgmm.fit(samples)

	# Denoise the image before segmenting.
	denoised = cv2.fastNlMeansDenoisingColored(
		img_small, 
		denoise_strength, 
		denoise_strength
	)

	# Categorize the pixels from the larger image.
	data    = denoised.reshape(
		denoised.shape[0] * denoised.shape[1], 
		denoised.shape[2]
	)
	classes = bgmm.predict(data)

	(values, counts) = np.unique(classes, return_counts=True)
	index            = np.argmax(counts)
	most_common      = values[index]
	bg_color         = torgb(colors[most_common])

	colored_img = denoised.copy()

	for c in range(n_classes):
		mask = (classes == c).reshape(denoised.shape[0], denoised.shape[1])
		colored_img[mask] = torgb(colors[c])

	# Dilate and then erode the colored image.
	#kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
	#dilated     = cv2.dilate(colored_img, kernel)
	#kernel2     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
	#colored_img = cv2.erode(dilated, kernel2)

	# Make a black border of one pixel around the image
	# so that the contouring algorithm won't mess up flakes
	# that are cut off. The color needs to be the same as
	# the background class. 
	colored_img = cv2.copyMakeBorder(
		colored_img, 
		border_width, 
		border_width, 
		border_width, 
		border_width, 
		cv2.BORDER_CONSTANT,
		value=bg_color
	)

	img_small = cv2.copyMakeBorder(
		img_small, 
		border_width, 
		border_width, 
		border_width, 
		border_width, 
		cv2.BORDER_CONSTANT,
		value=[0, 0, 0]
	)

	plotimg(img_small,   231, "Image")
	plotimg(denoised,    232, "Denoised")
	plotimg(colored_img, 233, "Segmented")

	

	# Now we run edge detectiong and contouring on the segmented image.
	edges = cv2.Canny(colored_img, 10, 80)

	# Dilate the edges to close close contours
	kernel  = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, 
		(edge_dilate_size, edge_dilate_size)
	)
	dilated = cv2.dilate(edges, kernel)

	contours, heirarchy = cv2.findContours(
		dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)

	contours = getLargerThan(
		contours, 
		(img_small.shape[0] * filter_edge_length)**2
	)


	# Draw the contours onto the original image.
	contoured = img_small.copy()
	for i in range(len(contours)):
		contoured = cv2.drawContours(
			contoured, 
			contours, 
			i, (0, 255, 0), 
			contour_thickness
		)

	# plotimg(edges,     234, "Edges")
	plotimg(dilated,   234, "Dilated Edges")
	plotimg(contoured, 235, "Contours")

	boxed = img_small.copy()
	# Draw rotated bounding rectangles around the flakes.
	for i in range(len(contours)):
		rect = cv2.minAreaRect(contours[i])
		box  = cv2.boxPoints(rect)
		box  = np.int0(box)
		boxed = cv2.drawContours(boxed, [box], 0, (255, 0, 0), 2)

	plotimg(boxed, 236, "Bounding Boxes")
		

	print("Contour Areas:")
	for c in contours:
		print('    %s'%str(cv2.contourArea(c)))

	plt.tight_layout()
	plt.show()

	# Show the largest contour.
	largest = None
	size    = 0
	for contour in contours:
		s = cv2.contourArea(contour)
		if s > size:
			largest = contour
			size    = s

	# Plot the flakes individually.
	rect0 = cv2.minAreaRect(largest)

	img0 = subimage(img_small, rect0)

	plotimg(img0, 121, "Largest")

	masked = img_small.copy()
	# Fill every pixel that isn't inside the contour in as black.
	mask = np.zeros((masked.shape[0], masked.shape[1]))
	mask = cv2.fillPoly(mask, [largest], 1)

	kernel  = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, 
		(mask_erode_size, mask_erode_size)
	)
	mask    = cv2.erode(mask, kernel)

	selection = mask == 0
	masked[selection] = [0, 0, 0]
	#code.interact(local=locals())

	img1 = subimage(masked, rect0)

	plotimg(img1, 122, "Masked")


	plt.tight_layout()
	plt.show()

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
		extract_flakes(os.path.join(args.directory, file))



















