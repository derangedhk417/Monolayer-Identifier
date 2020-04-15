# Description: This file contains the class that takes a microscope image
# and extracts all of the flakes that it can from the image. It is designed
# so that it can export images of each step in the process so that a user
# can adjust parameters that effect the process as a "calibration" step. 

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
from colors                import dist_colors  as colors

def torgb(h):
	return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]

class FlakeDifferentiator:
	def __init__(self, **kwargs):
		self.bgmm = None
		self.mode = 'HSV' # Options are 'HSV', 'RGB'

		# If the user specifies a BGMM, we will use that.
		# Otherwise, they'll have to call the fit function
		# and pass in one or more images to use for the fitting
		# process.
		if 'bgmm' in kwargs:
			self.bgmm = kwargs['bgmm']

		# Make sure they specified a valid color mode.
		if 'mode' in kwargs:
			self.mode = kwargs['mode']
			if self.mode not in ['HSV', 'RGB']:
				raise ValueError("""
					'mode' must be 'HSV' or 'RGB'
				""")

		# Now we setup the default parameters that control how
		# the images are processed for flake extraction.
		self.params = {
			'training_image_width'  : 100,
			'processed_image_width' : 1000,
			'denoising_strength'    : 30,
			'bgmm_classes'          : 3,
			'border_width'          : 5,
			'edge_threshold1'       : 10,
			'edge_threshold2'       : 80,
			'edge_dilate_size'      : 3,
			'length_threshold'      : 0.02,
			'mask_erode_size'       : 12
		}

		# These jsut effect how the steps in the process are displayed.
		self.ui_options = {
			'contour_color'     : (255, 0, 0),
			'bounds_color'      : (255, 0, 0),
			'contour_thickness' : 1,
			'bounds_thickness'  : 1,
			'mask_color'        : (0, 0, 0)
		}

	# Fits the BGMM used to identify flakes, using the specified 
	# images.
	#
	# Parameters:
	#      - images: A list of image file paths.
	def fit(self, images):
		# Load the images off of the disk.
		if self.mode == 'HSV':
			images = [cv2.imread(f, cv2.IMREAD_COLOR) for f in images]
			images = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in images]
		else:
			images = [cv2.imread(f, cv2.IMREAD_COLOR) for f in images]
			images = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images]

		# Downscale the images.
		tmp = []
		for img in images:
			width = self.params['training_image_width']
			scale = np.sqrt((width**2) / (img.shape[0] * img.shape[1]))
			tmp.append(cv2.resize(img, (0, 0), fx=scale, fy=scale))

		images = tmp

		# Convert the images into a set of HSV or RGB values
		# (depending on the mode) that will be used to fit the
		# BGMM. Shape will be (n_pixels, 3)

		self.fitting_data = images[0].reshape(
			images[0].shape[0] * images[0].shape[1],
			3
		)

		for img in images[1:]:
			additional_data = img.reshape(
				img.shape[0] * img.shape[1],
				3
			)
			self.fitting_data = np.concatenate((
				self.fitting_data, additional_data
			))

		self.bgmm = BayesianGaussianMixture(
			n_components=self.params["bgmm_classes"],
			covariance_type='full'
		)

		self.bgmm.fit(self.fitting_data)


	def getLargerThan(self, contours, s):
		selected = []

		for c in contours:
			area = cv2.contourArea(c)
			if area > s:
				selected.append(c)

		return selected

	def subimage(self, image, rect):
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

	# Loads the specified image file off of the disk and return the
	# following:
	#      1) A list of images that show parts of the process.
	#      2) A list of bounding rectangles that can be used to
	#         create subimages of flakes found during the process.
	#      3) A list of masks that differentiate the parts of the
	#         subimages that contain the flake from parts that dont.
	def scanImage(self, path):
		img = cv2.imread(path, cv2.IMREAD_COLOR)

		if self.mode == 'HSV':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Scale the image down to the requested size.
		if self.params['processed_image_width'] != 0:
			s     = self.params['processed_image_width']
			scale = np.sqrt((s**2) / (img.shape[0] * img.shape[1]))
			img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

		# Denoise the image.
		denoise_strength = self.params['denoising_strength']
		if denoise_strength != 0:
			denoised = cv2.fastNlMeansDenoisingColored(
				img, 
				denoise_strength, 
				denoise_strength
			)
		else:
			denoised = img

		# Segment the image using the bgmm.
		data = denoised.reshape(
			denoised.shape[0] * denoised.shape[1], 
			denoised.shape[2]
		)

		classes = self.bgmm.predict(data)

		(values, counts) = np.unique(classes, return_counts=True)
		index            = np.argmax(counts)
		most_common      = values[index]
		bg_color         = torgb(colors[most_common])

		colored_img = denoised.copy()

		for c in range(self.params['bgmm_classes']):
			mask = (classes == c).reshape(
				denoised.shape[0], denoised.shape[1]
			)
			colored_img[mask] = torgb(colors[c])


		# Make a black border of one pixel around the image
		# so that the contouring algorithm won't mess up flakes
		# that are cut off. The color needs to be the same as
		# the background class. 
		colored_img = cv2.copyMakeBorder(
			colored_img, 
			self.params['border_width'], 
			self.params['border_width'], 
			self.params['border_width'], 
			self.params['border_width'], 
			cv2.BORDER_CONSTANT,
			value=bg_color
		)

		img = cv2.copyMakeBorder(
			img, 
			self.params['border_width'], 
			self.params['border_width'], 
			self.params['border_width'], 
			self.params['border_width'], 
			cv2.BORDER_CONSTANT,
			value=[0, 0, 0]
		)

		# Perform the edge detection.
		edges = cv2.Canny(
			colored_img, 
			self.params['edge_threshold1'], 
			self.params['edge_threshold2']
		)

		# Dilate the edges to close close contours
		dilate_size = self.params['edge_dilate_size']
		if dilate_size != 0:
			kernel  = cv2.getStructuringElement(
				cv2.MORPH_ELLIPSE, 
				(dilate_size, dilate_size)
			)
			dilated = cv2.dilate(edges, kernel)
		else:
			dilated = edges

		# Find the contours. This will extract only the outer
		# most contours.
		contours, heirarchy = cv2.findContours(
			dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)

		# Filter out the contours that are smaller than the threshold
		# parameter given.
		contours = self.getLargerThan(
			contours, 
			(img.shape[0] * self.params['length_threshold'])**2
		)

		# Draw the contours onto the original image.
		contoured = img.copy()
		for i in range(len(contours)):
			contoured = cv2.drawContours(
				contoured, 
				contours, 
				i, self.ui_options['contour_color'], 
				self.ui_options['contour_thickness']
			)


		# Draw an image with bounding rectangles around the flakes.
		boxed      = img.copy()
		rectangles = []
		for i in range(len(contours)):
			rect  = cv2.minAreaRect(contours[i])
			rectangles.append(rect)
			box   = cv2.boxPoints(rect)
			box   = np.int0(box)
			boxed = cv2.drawContours(
				boxed, 
				[box], 
				0, 
				self.ui_options['bounds_color'], 
				self.ui_options['bounds_thickness']
			)

		
		masks = []
		for contour in contours:
			mask = np.zeros((img.shape[0], img.shape[1]))
			mask = cv2.fillPoly(mask, [contour], 1)

			# Erode the mask so that it will fit the flake more tightly.
			erode_size = self.params['mask_erode_size']
			if erode_size != 0:
				kernel  = cv2.getStructuringElement(
					cv2.MORPH_ELLIPSE, 
					(erode_size, erode_size)
				)
				mask = cv2.erode(mask, kernel)
			

			selection = mask == 0

			masks.append(selection)

		return (
			(img, denoised, colored_img, dilated, contoured, boxed), 
			rectangles, 
			masks
		)




















