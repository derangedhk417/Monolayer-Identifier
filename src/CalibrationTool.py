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
from  FlakeDifferentiator import FlakeDifferentiator
from matplotlib.widgets import Slider, Button, RadioButtons

def torgb(h):
	return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]


# Processes the arguments, sets up directories and validates that files
# actually exist.
def preprocess():
	parser = argparse.ArgumentParser(
		description='Adjust parameters of the flake extraction process.'
	)

	parser.add_argument(
		'-o', '--output', dest='output', type=str, required=True,
		help='The output file to write the data to.'
	)

	parser.add_argument(
		'-d', '--directory', dest='directory', type=str, required=True,
		help='The directory to read images from.'
	)

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	args = preprocess()


	# The files to read.
	files = [f for f in os.listdir(args.directory)]

	# Filter to only actual files.
	tmp = []
	for f in files:
		if os.path.isfile(os.path.join(args.directory, f)):
			tmp.append(f)

	files = sorted(tmp)
	files = [os.path.join(args.directory, file) for file in files]

	# Load the first three images and pass them to the BGMM trainer.
	diff = FlakeDifferentiator(mode='HSV')

	diff.params['processed_image_width'] = 350

	print("Fitting BGMM")
	diff.fit(files[0:1])
	print("Done Fitting")

	current_idx = 0

	images, rects, masks = diff.scanImage(files[0])

	# Draw the largest flake, zoomed in and masked.
	largest      = 0
	largest_rect = None
	largest_mask = None

	for rect, mask in zip(rects, masks):
		size = rect[1][0] * rect[1][1]

		if size > largest:
			largest      = size
			largest_rect = rect
			largest_mask = mask

	fig, (row1, row2) = plt.subplots(2, 4)
	(base, denoise, segment, edge) = row1
	(cont, box, zoom, masked)      = row2

	base.set_xticks([])
	base.set_yticks([])
	denoise.set_xticks([])
	denoise.set_yticks([])
	segment.set_xticks([])
	segment.set_yticks([])
	edge.set_xticks([])
	edge.set_yticks([])
	cont.set_xticks([])
	cont.set_yticks([])
	box.set_xticks([])
	box.set_yticks([])
	zoom.set_xticks([])
	zoom.set_yticks([])
	masked.set_xticks([])
	masked.set_yticks([])


	base_img = base.imshow(images[0])
	base.set_title("Image")

	denoise_img = denoise.imshow(images[1])
	denoise.set_title("Denoised")

	segment_img = segment.imshow(images[2])
	segment.set_title("Segmented")

	edge_img = edge.imshow(images[3])
	edge.set_title("Dilated Edges")

	cont_img = cont.imshow(images[4])
	cont.set_title("Contours")

	box_img = box.imshow(images[5])
	box.set_title("Bounds")

	img_largest = diff.subimage(images[0], largest_rect)
	img_masked  = images[0].copy()
	img_masked[largest_mask] = [0, 0, 0]
	img_masked  = diff.subimage(img_masked, largest_rect)

	zoom_img = zoom.imshow(img_largest)
	zoom.set_title("Largest Flake")

	mask_img = masked.imshow(img_masked)
	masked.set_title("Masked")

	plt.tight_layout()
	plt.subplots_adjust(
		bottom=0.15, top=1.05, hspace=0.0, left=0.02, right=0.98
	)

	def loadImage(idx):
		images, rects, masks = diff.scanImage(files[idx])

		# Draw the largest flake, zoomed in and masked.
		largest      = 0
		largest_rect = None
		largest_mask = None

		for rect, mask in zip(rects, masks):
			size = rect[1][0] * rect[1][1]

			if size > largest:
				largest      = size
				largest_rect = rect
				largest_mask = mask

		base_img.set_data(images[0])
		denoise_img.set_data(images[1])
		segment_img.set_data(images[2])
		edge_img.set_data(images[3])
		cont_img.set_data(images[4])
		box_img.set_data(images[5])

		img_largest = diff.subimage(images[0], largest_rect)
		img_masked  = images[0].copy()
		img_masked[largest_mask] = [0, 0, 0]
		img_masked  = diff.subimage(img_masked, largest_rect)

		zoom_img.set_data(img_largest)
		mask_img.set_data(img_masked)


		plt.draw()


	def _next(event):
		global current_idx
		if current_idx == len(files) - 1:
			current_idx = 0
		else:
			current_idx += 1

		loadImage(current_idx)

	def _prev(event):
		global current_idx
		if current_idx == 0:
			current_idx = len(files) - 1
		else:
			current_idx -= 1

		loadImage(current_idx)

	ax_next_button = plt.axes([0.11, 0.01, 0.05, 0.05])
	ax_prev_button = plt.axes([0.05, 0.01, 0.05, 0.05])

	next_button = Button(ax_next_button, "Next")
	next_button.on_clicked(_next)
	prev_button = Button(ax_prev_button, "Prev")
	prev_button.on_clicked(_prev)

	ax_denoise_strength = plt.axes([0.12, 0.07, 0.82, 0.015])
	sl_denoise_strength = Slider(
		ax_denoise_strength, 'Denoise Strength',
		0, 60, valinit=30, valfmt='%d', valstep=2,
		dragging=False
	)

	ax_edge_th1_strength = plt.axes([0.12, 0.087, 0.82, 0.015])
	sl_edge_th1_strength = Slider(
		ax_edge_th1_strength, 'Threshold 1',
		2, 120, valinit=10, valfmt='%d', valstep=5,
		dragging=False
	)

	ax_edge_th2_strength = plt.axes([0.12, 0.104, 0.82, 0.015])
	sl_edge_th2_strength = Slider(
		ax_edge_th2_strength, 'Threshold 2',
		2, 250, valinit=80, valfmt='%d', valstep=5,
		dragging=False
	)

	ax_dilate_size = plt.axes([0.12, 0.121, 0.82, 0.015])
	sl_dilate_size = Slider(
		ax_dilate_size, 'Edge Dilate',
		0, 8, valinit=3, valfmt='%d', valstep=1,
		dragging=False
	)

	ax_length_threshold = plt.axes([0.12, 0.138, 0.82, 0.015])
	sl_length_threshold = Slider(
		ax_length_threshold, 'Min Flake Width',
		0.00, 0.1, valinit=0.02, valfmt='%1.3f',
		dragging=False
	)

	ax_mask_erode = plt.axes([0.12, 0.155, 0.82, 0.015])
	sl_mask_erode = Slider(
		ax_mask_erode, 'Mask Erode Size',
		0, 30, valinit=12, valfmt='%d', valstep=1,
		dragging=False
	)

	ax_radio = plt.axes([0.19, 0.01, 0.2, 0.05])
	radio    = RadioButtons(ax_radio, ('HSV', 'RGB'), active=0)

	def set_mode(label):
		diff.mode = label
		diff.fit(files[0:1])
		# We need to retrain the BGMM for the new color space
		loadImage(current_idx)

	radio.on_clicked(set_mode)

	def update_denoise_strength(val):
		diff.params['denoising_strength'] = int(val)
		loadImage(current_idx)

	sl_denoise_strength.on_changed(update_denoise_strength)

	def update_th1(val):
		diff.params['edge_threshold1'] = int(val)
		loadImage(current_idx)

	sl_edge_th1_strength.on_changed(update_th1)

	def update_th2(val):
		diff.params['edge_threshold2'] = int(val)
		loadImage(current_idx)

	sl_edge_th2_strength.on_changed(update_th2)

	def update_dilate_size(val):
		diff.params['edge_dilate_size'] = int(val)
		loadImage(current_idx)

	sl_dilate_size.on_changed(update_dilate_size)

	def update_length_threshold(val):
		diff.params['length_threshold'] = float(val)
		loadImage(current_idx)

	sl_length_threshold.on_changed(update_length_threshold)

	def update_mask_erode(val):
		diff.params['mask_erode_size'] = int(val)
		loadImage(current_idx)

	sl_mask_erode.on_changed(update_mask_erode)
		


	plt.show()















