import socket
import cv2
import numpy as np
import math

class RCControl(object):

	def steer(self, prediction):
		if prediction == 2:
			print("Forward")
		elif prediction == 0:
			print("Left")
		elif prediction == 1:
			print("Right")
		else:
			self.stop()

	def stop(self):
		print('Stop')

class NeuralNetwork(object):

	def __init__(self):	
		self.model = cv2.ml.ANN_MLP_create()

	def create(self):
		layer_sizes = np.int32([38400, 72, 4])
		self.model.setLayerSizes(layer_sizes)
		self.model.load('mlp_xml/mlp.xml')

	def predict(self, samples):
		ret, resp = self.model.predict(samples)
		return resp.argmax(-1)




class StreamingServer(object):
	def __init__(self):
		self.restUrl = 'http://192.168.1.106:8080/control'
		# self.restUrl = 'http://192.168.1.106:5000/messages'
		self.server_socket = socket.socket()
		self.server_socket.bind(('192.168.1.102', 8000))
		self.server_socket.listen(1)
		self.conn, self.client_address = self.server_socket.accept()
		self.connection = self.conn.makefile('rb')

		self.model = NeuralNetwork()
		self.model.create()

		self.rc_car = RCControl()
		self.streamingAndCollectData()

	def streamingAndCollectData(self):
		# collect images for training
		print('Start collecting images...')
		e1 = cv2.getTickCount()
		image_array = np.zeros((1, 38400))
		label_array = np.zeros((1, 4), 'float')

		try:
			print("Connection from: ", self.client_address)
			print("Streaming...")
			print("Press 'Q' to exit")

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
					gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
					image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)

					# lower half of the image
					half_gray = gray[120:240, :]

					cv2.imshow('image', image)
					# reshape image
					image_array = half_gray.reshape(1, 38400).astype(np.float32)

					# neural network makes prediction
					prediction = self.model.predict(image_array)

					self.rc_car.steer(prediction)

					if cv2.waitKey(1) == 27:
						exit(0)
					

			
		finally:
			self.connection.close()
			self.server_socket.close()

if __name__ == '__main__':
	StreamingServer()