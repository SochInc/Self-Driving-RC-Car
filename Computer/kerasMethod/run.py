import numpy as np
import cv2
import socket
from getKeys import key_check
import requests
import keras.models

class StreamingServer(object):
	def __init__(self):
		self.restUrl = 'http://192.168.0.106:5000/messages'
		self.server_socket = socket.socket()
		self.server_socket.bind(('192.168.0.110', 8000))
		self.server_socket.listen(1)
		self.conn, self.client_address = self.server_socket.accept()
		self.connection = self.conn.makefile('rb')

		# Taking model instance from keras
		self.model = keras.models.load_model('nn_h5/nn.h5')

		self.streamingAndCollectData()

	def preprocess(self, frame):
		image_array = frame.reshape(1, 38400).astype(np.float32)
		image_array = image_array / 255.
		return image_array

	def predict(self, image):
		image_array = self.preprocess(image)
		y_hat       = self.model.predict(image_array)
		i_max       = np.argmax(y_hat)
		y_hat_final = np.zeros((1,4), 'float')
		np.put(y_hat_final, i_max, 1)
		return y_hat_final[0]

	def streamingAndCollectData(self):
		saved_frame = 0
		total_frame = 0
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

					# Predict the data based on image frame received.
					prediction = self.predict(roi)

					# Control based on prediction
					self.navigate(prediction)
	
					# Press Q to Quit runnuing the car..
					keys = key_check()
                    if 'Q' in keys:
						payload = dict(data='x')
						response = requests.post(self.restUrl, params=payload)
						print(response, payload, 'sent to server.')
						exit(0)

		finally:
			self.connection.close()
			self.server_socket.close()
	
	def navigate(self, prediction):
		print(prediction)
		if np.all(prediction   == [ 0., 0., 1., 0.]):
			print('forward')
			payload = dict(data='s')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')

		elif np.all(prediction == [ 1., 0., 0., 0.]):
			print('left')
			payload = dict(data='a')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')

		elif np.all(prediction == [ 0., 1., 0., 0.]):
			print('right')
			payload = dict(data='d')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')
			
		elif np.all(prediction == [ 0., 0., 0., 1.]):
			print('reverse')
			payload = dict(data='w')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')

if __name__ == '__main__':
	StreamingServer()