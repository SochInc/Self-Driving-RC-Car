import numpy as np
import cv2
import socket
import requests
import json
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

class StreamingServer(object):
	def __init__(self):
		self.restUrl = 'http://192.168.0.106:5000/messages'
		self.server_socket = socket.socket()
		self.server_socket.bind(('192.168.0.104', 8000))
		self.server_socket.listen(1)
		self.conn, self.client_address = self.server_socket.accept()
		self.connection = self.conn.makefile('rb')
		self.streamingAndCollectData()	

	def predict(self, image):
		with open('model.json','r') as jfile:
			model = model_from_json(jfile.read())
		model.compile("adam", "mse")
		weights_file = 'model.h5'
		model.load_weights(weights_file)
		return model.predict(image, batch_size=1)

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
					transformed_image_array = image[None, :, :, :]

					# Predict the data based on image frame received.
					prediction = self.predict(transformed_image_array)

					# Control based on prediction
					self.navigate(int(prediction))
					print(int(prediction))
		finally:
			self.connection.close()
			self.server_socket.close()
	
	def navigate(self, prediction):
		print(prediction)
		if prediction  == 0:
			print('forward')
			payload = dict(data='w')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')

		elif prediction == 3:
			print('left')
			payload = dict(data='a')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')

		elif prediction == 2:
			print('right')
			payload = dict(data='d')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')
			
		elif prediction == 4:
			print('reverse')
			payload = dict(data='s')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')
		elif prediction == 1:
			print('release')
			payload = dict(data='x')
			response = requests.post(self.restUrl, params=payload)
			print(response, payload, 'sent to server.')
if __name__ == '__main__':
	StreamingServer()