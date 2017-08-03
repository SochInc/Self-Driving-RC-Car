import numpy as np
import cv2
import socket
import glob
import os

list = os.listdir('training_images')
number_images = len(list)

image_array = np.zeros((1, 2400))
#86400 is the image size
label_array = np.zeros((1, 4), 'float')
def auto_canny(img, sigma=0.01):
	# compute the median of the single channel pixel intensities
	v = np.median(img)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)

	# return the edged image
	return edged

def draw_edges(img, n_components=-1):
	"""
	Accepts an image, draws edges on the image, and returns the edged image.
	:param img: Image to be drawn on
	:return: Image of edges only
	"""
	edges = auto_canny(img)
	contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	if n_components != -1 and type(n_components) == int:
	    contours = contours[:n_components]
	elif n_components != -1 and type(n_components) == float:
	    contours = contours[:int(len(contours) * n_components)]
	img = np.zeros(img.shape)
	cv2.drawContours(img, contours, -1, 255, 1)
	return img

training_data = glob.glob('training_data_temp/*.npz')
for single_npz in training_data:
	i = 1
	j = 0
	with np.load(single_npz) as datas:
		for data in datas['train']:
			i1 = i*10-4
			i2 = i1+1
			i3 = i2+1
			i4 = i3+1
			i5 = i4+1
			i += 1
			if i1 > number_images:
				break
			img1 = cv2.resize(cv2.imread('training_images/Imageframe{:>05}.jpg'.format(i1),0), (0,0), fx=0.25, fy=0.25)
			img1 = draw_edges(img1, 0.05).flatten().astype(np.float32)
			img2 = cv2.resize(cv2.imread('training_images/Imageframe{:>05}.jpg'.format(i2),0), (0,0), fx=0.25, fy=0.25)
			img2 = draw_edges(img2, 0.05).flatten().astype(np.float32)
			img3 = cv2.resize(cv2.imread('training_images/Imageframe{:>05}.jpg'.format(i3),0), (0,0), fx=0.25, fy=0.25)
			img3 = draw_edges(img3, 0.05).flatten().astype(np.float32)
			img4 = cv2.resize(cv2.imread('training_images/Imageframe{:>05}.jpg'.format(i4),0), (0,0), fx=0.25, fy=0.25)
			img4 = draw_edges(img4, 0.05).flatten().astype(np.float32)
			img5 = cv2.resize(cv2.imread('training_images/Imageframe{:>05}.jpg'.format(i5),0), (0,0), fx=0.25, fy=0.25)
			img5 = draw_edges(img5, 0.05).flatten().astype(np.float32)
			image_array = np.vstack((image_array, img1))
			image_array = np.vstack((image_array, img2))
			image_array = np.vstack((image_array, img3))
			image_array = np.vstack((image_array, img4))
			image_array = np.vstack((image_array, img5))
			train_labels_temp = datas['train_labels'][j]
			j+=1
			label_array = np.vstack((label_array, train_labels_temp))
			label_array = np.vstack((label_array, train_labels_temp))
			label_array = np.vstack((label_array, train_labels_temp))
			label_array = np.vstack((label_array, train_labels_temp))
			label_array = np.vstack((label_array, train_labels_temp))
			#train_temp, train_labels_temp = unison_shuffled_copies(train_temp, train_labels_temp)

print(image_array.shape)
print(label_array.shape)


train = np.array(image_array[1:, :], dtype=np.float32)
train_labels = np.array(label_array[1:, :], dtype=np.float32)

np.savez('training_data_updated/test.npz', train=train, train_labels=train_labels)