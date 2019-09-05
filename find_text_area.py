from __future__ import print_function

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import style
import cv2
from sklearn.cluster import DBSCAN
import utils



def find_text_area(path_original_page):
	final_img_color = cv2.imread(path_original_page)
	img = cv2.imread(path_original_page, cv2.IMREAD_GRAYSCALE)
	ret3, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	thresh_img = cv2.medianBlur(thresh_img, 3)
	thresh_img = utils.remove_small_component(thresh_img,6)

	contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	x_value = []
	y_value = []
	contours = contours[1:]
	for contour in contours:
		[x, y, w, h] = cv2.boundingRect(contour)
		x_value.append(x)
		x_value.append(x + w)
		y_value.append(y)
		y_value.append(y + h)

	X = np.array(list(zip(x_value, y_value)))

	clustering = DBSCAN(eps=45, metric='euclidean').fit(X)

	y_pred = clustering.fit_predict(X)

	(values, counts) = np.unique(y_pred, return_counts=True)

	ind = np.argmax(counts)
	label_text = values[ind]

	coordinates_x_text = []
	coordinates_y_text = []

	for i in range(len(X)):
		if y_pred[i] == label_text:
			coordinates_x_text.append(x_value[i])
			coordinates_y_text.append(y_value[i])

	x_min = min(coordinates_x_text)
	x_max = max(coordinates_x_text)
	y_min = min(coordinates_y_text)
	y_max = max(coordinates_y_text)

	if x_min > 5:
		text_area_img_binary = thresh_img[y_min - 15:y_max + 15, x_min - 5:x_max + 5]
		final_img_color = final_img_color[y_min - 15:y_max + 15, x_min - 5:x_max + 5]

	else:
		text_area_img_binary = thresh_img[y_min - 15:y_max + 15, x_min:x_max + 5]
		final_img_color = final_img_color[y_min - 15:y_max + 15, x_min:x_max + 5]
	#########################
	### PER RISOLVERE PROBLEMA TROPPO TESTO
	black_px_per_column = np.sum(text_area_img_binary == 0, axis=0)
	black_px_per_column = utils.smooth(black_px_per_column, 5, 'flat')

	avg = (np.average(black_px_per_column) * 0.60)

	black = []
	for i in black_px_per_column:
		black.append(int(i))
	ixs = np.argwhere(black < avg)
	index = []
	for i in ixs:
		index.append(i[0])
	values = utils.group_consecutive_values(index, 5)

	new_thres = []
	for i in values:
		if (len(i)) > 90:
			new_thres.append((min(i), max(i)))

	if (len(new_thres) == 1):
		value_thres = new_thres[0][1]
		if (value_thres < text_area_img_binary.shape[1] - value_thres):
			final_img_color = final_img_color[0:final_img_color.shape[0], new_thres[0][1] - 60:final_img_color.shape[1]]
		else:
			final_img_color = final_img_color[0:final_img_color.shape[0], 0:new_thres[0][0] + 120]

	elif (len(new_thres) == 2):
		sorted_th = sorted(new_thres, key=lambda tup: tup[0])
		# print(sorted)
		final_img_color = final_img_color[0:final_img_color.shape[0], sorted_th[0][1] - 60:final_img_color.shape[1]]
		final_img_color = final_img_color[0:final_img_color.shape[0], 0:sorted_th[1][0] - (sorted_th[0][1] - 60) + 12]

	final_img_color = cv2.fastNlMeansDenoising(final_img_color, None, 20, 7, 21)
	return final_img_color



