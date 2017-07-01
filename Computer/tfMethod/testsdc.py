import numpy as np
import cv2
import socket
import time
from getKeys import key_check
import requests
import urllib3
import json
import os
from models import pinet

WIDTH = 320
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'trained_models/SDCModel-{}-{}-{}-epochs-300K-data.model'.format(LR, 'pinetv2',EPOCHS)
t_time = 0.09

class StreamingServer(object):
	def __init__(self):
		# Control keys value
		self.restUrl = 'http://192.168.1.106:8080/control'
		self.server_socket = socket.socket()
		self.server_socket.bind(('192.168.1.102', 8000))
		self.server_socket.listen(1)
		self.conn, self.client_address = self.server_socket.accept()
		self.connection = self.conn.makefile('rb')
	   
		self.streamingAndCollectData()

	def streamingAndCollectData(self):
		model = pinet (WIDTH, HEIGHT, LR)
		model.load(MODEL_NAME)

		print('Start Testing self driving car.')
		e1 = cv2.getTickCount()

		for i in list(range(4))[::-1]:
			print(i+1)
			time.sleep(1)

		last_time = time.time()
		print('STARTING!!!')

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
					image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

					# select lower half of the image
					roi = image[120:240, :]

					last_time = time.time()
					# run a color convert:
					screen = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
					screen = cv2.resize(screen, (320,120))
					cv2.imshow('image', screen)

					prediction = model.predict([screen.reshape(320,120,1)])[0]
					print(prediction)

					turn_thresh = .75
					fwd_thresh = 0.70

					if prediction[1] > fwd_thresh:
						print('Forward')
					elif prediction[0] > turn_thresh:
						print('left')
					elif prediction[2] > turn_thresh:
						print('right')
					else:
						print('Forward')

			e2 = cv2.getTickCount()
			# calculate streaming duration
			time0 = (e2 - e1) / cv2.getTickFrequency()
			print('Streaming duration:', time0)
			
		finally:
			self.connection.close()
			self.server_socket.close()

		

if __name__ == '__main__':
	StreamingServer()