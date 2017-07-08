import numpy as np
import cv2
import socket
from getKeys import key_check
import requests
import urllib3
import json

from sklearn.externals import joblib
from datetime import datetime as dt

log_model = joblib.load('result/logistic_regression.pkl')

class StreamingServer(object):
	def __init__(self):
		self.server_socket = socket.socket()
		self.server_socket.bind(('192.168.1.102', 8000))
		self.server_socket.listen(1)
		self.conn, self.client_address = self.server_socket.accept()
		self.connection = self.conn.makefile('rb')

		# create labels
		self.k = np.zeros((4, 4), 'float')
		for i in range(4):
			self.k[i, i] = 1
		self.temp_label = np.zeros((1, 4), 'float')
		self.streamingAndCollectData()

	def streamingAndCollectData(self):
		saved_frame = 0
		total_frame = 0
		
		# collect images for training
		print('Start collecting images...')
		e1 = cv2.getTickCount()

		try:
			print("Connection from: ", self.client_address)
			print("Streaming...")

			stream_bytes = b''
			frame = 1
			while True:
				stream_bytes += self.connection.read(1024)
				first = stream_bytes.find(b'\xff\xd8')
				last = stream_bytes.find(b'\xff\xd9')
				self.conn.sendall(b'WA')
				if first != -1 and last != -1:
					jpg = stream_bytes[first:last + 2]
					stream_bytes = stream_bytes[last + 2:]
					image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					height, width = image.shape
					# select lower half of the image
					roi = image[int(height/2):height, :]

					self.navigate(roi)
					if cv2.waitKey(1) == 27:
						exit(0)

		finally:
			self.connection.close()
			self.server_socket.close()
	
	def get_direction_from_image(self, img):
		"""
		Takes an image, converts it to black and white, resizes it to 1/4 of each original
		dimension (1/16th original size in total), and uses a logistic regression classification 
		to predict the correct direction.
		"""
		global log_model
		img_bw = img
		#img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		height, width = img_bw.shape
		img_bw = img_bw[int(height/2):height, :]
		img_bw = cv2.resize(img_bw, (0,0), fx=0.25, fy=0.5)
		img_bw = self.draw_edges_bw(img_bw, 0.05)
		img_bw_copy = img_bw.flatten().astype(np.float32)
		print(img_bw_copy.shape)
		prediction = log_model.predict([img_bw_copy])
		return prediction[0], img_bw

	def navigate(self, img):
		"""
		Takes in an image, gets the direction, and issues a command
		to drive the car.
		"""
		direction, img = self.get_direction_from_image(img)

		if direction == 2:
			print('forward')
		elif direction == 0:
			print('left')
		elif direction == 1:
			print('right')
		elif direction == 3:
			print('reverse')

		return img

	def draw_edges_bw(self, img, n_components=-1):
		"""
		Draws edges in black and white.
		"""
		contours = self.draw_edges(img, n_components)
		img_bw = np.zeros(img.shape)
		cv2.drawContours(img_bw, contours, -1, 255, 1)
		return img_bw

	def draw_edges(self, img, n_components=-1):
		"""
		Accepts an image, draws edges on the image, and returns the edged image.
		:param img: Image to be drawn on
		:return: Image of edges only
		"""
		edges = self.auto_canny(img)
		contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		if n_components != -1 and type(n_components) == int:
			contours = contours[:n_components]
		elif n_components != -1 and type(n_components) == float:
			contours = contours[:int(len(contours) * n_components)]
		cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
		return contours

	def auto_canny(self, img, sigma=0.01):
		"""
		Applies the Canny detection algorithm with automatic parameter estimation. Modified from:
		http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
		:param img: Image in
		:param sigma: A hyperparameter to tune how tight (small) the thresholds are
		:return: An image of edges
		"""
		#
		# compute the median of the single channel pixel intensities
		v = np.median(img)

		# apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(img, lower, upper)

		# return the edged image
		return edged

if __name__ == '__main__':
	StreamingServer()