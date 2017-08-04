import numpy as np
import cv2
import socket
from getKeys import key_check
import requests
import pygame
import os
import time

class StreamingServer(object):
    def __init__(self):
        # REST API url
        self.restUrl = 'http://192.168.0.106:5000/messages'

        # Start Socket Server
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.0.110', 8000))
        self.server_socket.listen(1)
        self.conn, self.client_address = self.server_socket.accept()
        self.connection = self.conn.makefile('rb')

        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')

        # Pygame Initialization
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((800,600))
        pygame.display.set_caption('SDRC')

        self.streamingAndCollectData()

    def streamingAndCollectData(self):
        saved_frame = 0
        total_frame = 0

        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        try:
            print("Connection from: ", self.client_address)
            print("Streaming Pi Camera...")
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
                    imageBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    height, width = imageBW.shape

                    # select lower half of the image
                    roi = imageBW[int(height/2):height, :]

                    # Show Image
                    cv2.imshow('image', image)
                                        
                    temp_array = roi.flatten().astype(np.float32)
                    frame += 1
                    total_frame += 1
                    
                    # Listen to pygame event
                    for event in pygame.event.get():
                        keystate = pygame.key.get_pressed()
                        if event.type == pygame.QUIT:
                            print("Quit")
                            break
                        if event.type == pygame.KEYDOWN:
                            print("Start Controlling...")
                            if keystate[pygame.K_UP]:
                                print("Forward")
                                cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), roi)
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))
                                saved_frame += 1

                                payload = dict(data='s')
                                response = requests.post(self.restUrl, params=payload)
                                print(response, payload, 'sent to server.')

                            elif keystate[pygame.K_DOWN]:
                                print("Reverse")
                                cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), roi)
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[3]))
                                saved_frame += 1

                                payload = dict(data='w')
                                response = requests.post(self.restUrl, params=payload)
                                print(response, payload, 'sent to server.')
                                        
                            elif keystate[pygame.K_LEFT]:
                                print("Left")
                                cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), roi)
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                saved_frame += 1

                                payload = dict(data='a')
                                response = requests.post(self.restUrl, params=payload)
                                print(response, payload, 'sent to server.')
                    
                            elif keystate[pygame.K_RIGHT]:
                                print("Right")
                                cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), roi)
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frame += 1

                                payload = dict(data='d')
                                response = requests.post(self.restUrl, params=payload)
                                print(response, payload, 'sent to server.')
                    
                            elif keystate[pygame.K_UP] and keystate[pygame.K_RIGHT]:
                                print("Forward Right")
                                cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), roi)
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frame += 1

                                payload = dict(data='q')
                                response = requests.post(self.restUrl, params=payload)
                                print(response, payload, 'sent to server.')

                            elif keystate[pygame.K_UP] and keystate[pygame.K_LEFT]:
                                print("Forward Left")
                                cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), roi)
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                saved_frame += 1

                                payload = dict(data='e')
                                response = requests.post(self.restUrl, params=payload)
                                print(response, payload, 'sent to server.')

                        if event.type == pygame.KEYUP:
                            print("release")
                            payload = dict(data='x')
                            response = requests.post(self.restUrl, params=payload)
                            print(response, payload, 'sent to server.')
            
                pygame.display.update()
                keys = key_check()
                if 'Q' in keys:
                        print('exit')
                        self.send_inst = False
                        break
            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]

            # save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:    
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)

            e2 = cv2.getTickCount()
            
            # calculate streaming metrics
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print('Streaming duration:', time0)

            print(train.shape)
            print(train_labels.shape)
            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)
            
        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == '__main__':
    StreamingServer()