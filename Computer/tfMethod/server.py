import numpy as np
import cv2
import socket
import time
from getKeys import key_check
import requests
import urllib3
import json
import os

class StreamingServer(object):
    def __init__(self):
        # Control keys value
        self.w = [1,0,0,0,0,0,0,0,0]
        self.s = [0,1,0,0,0,0,0,0,0]
        self.a = [0,0,1,0,0,0,0,0,0]
        self.d = [0,0,0,1,0,0,0,0,0]
        self.wa = [0,0,0,0,1,0,0,0,0]
        self.wd = [0,0,0,0,0,1,0,0,0]
        self.sa = [0,0,0,0,0,0,1,0,0]
        self.sd = [0,0,0,0,0,0,0,1,0]
        self.nk = [0,0,0,0,0,0,0,0,1]

        self.restUrl = 'http://192.168.1.106:8080/control'
        #self.restUrl = 'http://192.168.1.106:5000/messages'
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

        self.starting_value = 1
        # Get the Training data file
        while True:
            self.file_name = 'training_data_temp/training_data-{}.npy'.format(self.starting_value)
            if os.path.isfile(self.file_name):
                print('File exists, moving along',self.starting_value)
                self.starting_value += 1
            else:
                print('File does not exist, starting fresh!',self.starting_value)
                break
        
        self.streamingAndCollectData()

    def streamingAndCollectData(self):
        saved_frame = 0
        total_frame = 0

        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount()

        file_name = self.file_name
        starting_value = self.starting_value
        training_data = []
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)

        last_time = time.time()
        paused = False
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
                    screen = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    cv2.imshow('image', screen)

                    cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), screen)

                    frame += 1
                    total_frame += 1

                    # Check key stroke while streaming
                    keys = key_check()

                    if 'W' in keys and 'A' in keys:
                        print("Forward Left")
                        training_data.append([screen,self.wa])
                        saved_frame += 1

                        # Send key to rest api
                        payload = dict(data='WA')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')

                    elif 'W' in keys and 'D' in keys:
                        print("Forward Right")
                        training_data.append([screen,self.wd])
                        saved_frame += 1

                        # Send key to rest api
                        payload = dict(data='WD')
                        headers = { 'content-type': 'application/json' }
                        response = requests.post(self.restUrl, data=json.dumps(payload), headers=headers )
                        print(response, payload, 'sent to server.')
                        

                    elif 'S' in keys and 'A' in keys:
                        print("Reverse Left")
                        training_data.append([screen,self.sa])
                        # Send key to rest api
                        payload = dict(data='SA')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')
                        

                    elif 'S' in keys and 'D' in keys:
                        print("Reverse Right")
                        training_data.append([screen,self.sd])
                        # Send key to rest api
                        payload = dict(data='SD')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')
                        

                    elif 'W' in keys:
                        print("Forward")
                        saved_frame += 1
                        training_data.append([screen,self.w])

                        # Send key to rest api
                        payload = dict(data='W')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')

                    elif 'S' in keys:
                        print("Reverse")
                        saved_frame += 1
                        training_data.append([screen,self.s])
                        
                        # Send key to rest api
                        payload = dict(data='S')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')

                    elif 'D' in keys:
                        print("Right")
                        saved_frame += 1
                        training_data.append([screen,self.d])
                        
                        # Send key to rest api
                        payload = dict(data='D')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')


                    elif 'A' in keys:
                        print("Left")
                        saved_frame += 1
                        training_data.append([screen,self.a])
                        
                        # Send key to rest api
                        payload = dict(data='A')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')

                    elif 'Q' in keys:
                        print('exit')
                        training_data.append([screen,self.nk])
                        self.send_inst = False
                        # Send key to rest api
                        payload = dict(data='Q')
                        response = requests.post(self.restUrl, data=payload)
                        print(response, payload, 'sent to server.')

                        break


            np.save(file_name,training_data)

            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print('Streaming duration:', time0)
            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)
            
        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == '__main__':
    StreamingServer()