import cv2
import numpy as np
from tqdm import tqdm
from utils import binarization
from random import randint
# from matplotlib import pyplot as plt


#TERZO PASSO DELLA BINARIZZAZIONE: MIGLIORAMENTO DEI CONTORNI
def find_words_add_8_binarization(binary_img,img_gray):
	img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
	img_gray = cv2.copyMakeBorder(img_gray, 5, 5, 5, 5, cv2.BORDER_REPLICATE)

	contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hull = []
	# calculate points for each contour
	for i in range(len(contours)):
		# creating convex hull object for each contour
		hull.append(cv2.convexHull(contours[i]))

	drawing = np.zeros((binary_img.shape[0], binary_img.shape[1], 1), np.uint8)

	# draw contours and hull points
	for i in range(1, len(contours)):

		color = (255)
		# draw ith contour
		cv2.drawContours(drawing, hull, i, color, -1)


	nlabels, labels = cv2.connectedComponents(drawing, connectivity=8)

	new_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 1), np.uint8)
	for i in tqdm(range(1, (nlabels))):
		img_component = np.uint8(np.where(labels == i, 255, 0))
		img_and = cv2.bitwise_and(img_component, cv2.bitwise_not(binary_img))
		[x, y, w, h] = cv2.boundingRect(img_and)
		if x > 2:
			x = x-1
		if y>2:
			y = y-1
		if x+w < img_and.shape[1]-2:
			w = w+2
		if y+h<img_and.shape[0]-2:
			h = h+2
		binary_word = img_and[y:y+h,x:x+w]
		gray_word = (img_gray[y:y+h,x:x+w])
		binary_word = binarization.add_8_simple(binary_word, gray_word)

		img_and[y:y+h,x:x+w] = binary_word


		new_img = cv2.bitwise_or(new_img, img_and)

	new_img = binarization.remove_small_component_and_abbreviations(new_img, 8)
	return new_img


def find_words(binary_img):

	contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hull = []
	# calculate points for each contour
	for i in range(len(contours)):
		# creating convex hull object for each contour
		hull.append(cv2.convexHull(contours[i]))

	drawing = np.zeros((binary_img.shape[0], binary_img.shape[1], 1), np.uint8)

	# draw contours and hull points
	for i in range(1, len(contours)):
		color = (255)
		# draw ith contour
		cv2.drawContours(drawing, hull, i, color, -1)

	all_rect_width = []
	nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(drawing, connectivity=8)
	for a in range(1, nlabels):
		x, y, w, h, _ = stats[a]
		all_rect_width.append(w)
	avg_width = (np.average(all_rect_width) * 3.5)

	new_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), np.uint8)

	for i in tqdm(range(1, (nlabels))):
		x, y, w, h, _ = stats[i]
		#se larghezza della convex hull è > media
		#spezzala nelle sue componenti connesse
		if w > avg_width:
			sub_convex_diveded = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), np.uint8)
			img_component = np.uint8(np.where(labels == i, 255, 0))
			img_and = cv2.bitwise_and(img_component, cv2.bitwise_not(binary_img))
			kernel = np.ones((2, 2), np.uint8)
			dilatation = cv2.dilate(img_and, kernel, iterations=1)
			nlabels_sub, labels_sub, stats_sub, _ = cv2.connectedComponentsWithStats(dilatation, connectivity=8)
			for j in range(1, nlabels_sub):
				sub_convex = np.uint8(np.where(labels_sub == j, 255, 0))
				result_and = cv2.bitwise_and(img_and, sub_convex)
				result_and = cv2.cvtColor(result_and, cv2.COLOR_GRAY2BGR)
				r = randint(1, 255)
				g = randint(1, 255)
				b = randint(1, 255)
				result_and = np.uint8(np.where(result_and == [255, 255, 255], [r - i, g + i, b], [0, 0, 0]))
				sub_convex_diveded = cv2.bitwise_or(sub_convex_diveded, result_and)

			new_img = cv2.bitwise_or(new_img, sub_convex_diveded)

		else:
			img_component = np.uint8(np.where(labels == i, 255, 0))
			img_and = cv2.bitwise_and(img_component, cv2.bitwise_not(binary_img))
			img_and = cv2.cvtColor(img_and, cv2.COLOR_GRAY2BGR)


			r = randint(1, 255)
			g = randint(1, 255)
			b = randint(1, 255)
			img_and = np.uint8(np.where(img_and == [255, 255, 255], [r - i, g + i, b], [0, 0, 0]))

			new_img = cv2.bitwise_or(new_img, img_and)


	return new_img