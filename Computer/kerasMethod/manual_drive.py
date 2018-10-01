import numpy as np
import cv2
import socket
import requests
import pygame
import os
import time
import pandas as pd
UP = LEFT = DOWN = RIGHT = ACCELERATE = DECELERATE = False

class StreamingServer(object):
    def __init__(self):
        # REST API url
        self.restUrl = 'http://192.168.1.106:5000/messages'

        # Start Socket Server
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.1.101', 8000))
        self.server_socket.listen(1)
        self.conn, self.client_address = self.server_socket.accept()
        self.connection = self.conn.makefile('rb')
        
        # Pygame Initialization
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((800,600))
        pygame.display.set_caption('SDRC')

        # Stream and collect data
        self.streamAndCollectData()  

    def get_keys(self):
        change = False
        stop = False
        key_to_global_name = {
            pygame.K_LEFT: 'LEFT',
            pygame.K_RIGHT: 'RIGHT',
            pygame.K_UP: 'UP',
            pygame.K_DOWN: 'DOWN',
            pygame.K_ESCAPE: 'QUIT',
            pygame.QUIT: 'QUIT'
        }

        for event in pygame.event.get():
            if event.type in {pygame.K_q, pygame.QUIT}:
                stop = True
            elif event.type in {pygame.KEYDOWN, pygame.KEYUP}:
                down = (event.type == pygame.KEYDOWN)
                change = (event.key in key_to_global_name)
                if event.key in key_to_global_name:
                    globals()[key_to_global_name[event.key]] = down
        return (UP, DOWN, LEFT, RIGHT, change, stop)
    
    def sendData(self,dat):       
        payload = dict(data=dat)
        response = requests.post(self.restUrl, params=payload)
    
    def streamAndCollectData(self):
        saved_frame = 0
        total_frame = 0
        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount()
        image_path = []
        label_array =[]
        #image_array = np.zeros((1, 38400))
        #label_array = np.zeros((1, 5), 'float')
        if not os.path.exists('training_images'):
                os.makedirs('training_images')
        try:
            print("Connection from: ", self.client_address)
            print("Streaming Pi Camera...")
            stream_bytes = b''
            frame = 1
            crashed = False
            command = 'x'
            label = 4
            while not crashed:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                self.conn.sendall(b'WA')
                
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    cvimage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    pygameimage = pygame.image.frombuffer(cvimage.tostring(),cvimage.shape[1::-1],"RGB")
                    print(cvimage.shape[1::-1])
                    frame, total_frame = frame+1, total_frame+1
                    up_key, down, left, right, change, stop = self.get_keys()
                    if stop:
                        print("stop")
                        crashed = True
                    if change:
                        print('change')
                        command = 'x'
                        label = 4
                        cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), cvimage)
                        image_path.append(frame)
                        label_array.append(label)
                        saved_frame+=1
                        
                        if up_key:
                            print('up')
                            command = 'w'
                            label = 0
                            cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), cvimage)
                            image_path.append(frame)
                            label_array.append(label)
                            saved_frame+=1
                            
                        if down:
                            print('down')
                            command = 's'
                            label = 1
                            cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), cvimage)
                            image_path.append(frame)
                            label_array.append(label)
                            saved_frame+=1
                            
                        if right:
                            print('right')
                            command = 'd'
                            label =2
                            cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), cvimage)
                            image_path.append(frame)
                            label_array.append(label)
                            saved_frame+=1
                            
                        if left:
                            print('left')
                            command = 'a'
                            label = 3
                            cv2.imwrite('training_images/Imageframe{:>05}.jpg'.format(frame), cvimage)
                            image_path.append(frame)
                            label_array.append(label)
                            saved_frame+=1
                    
                    
                    self.sendData(command)

                    pygameimage = pygame.transform.scale(pygameimage, (800,600))
                    self.gameDisplay.fill((0,0,0))
                    self.gameDisplay.blit(pygameimage, (0, 0))
                    pygame.display.update()

            df = pd.DataFrame({"image_path":image_path,"labels":label_array})
            df.to_csv("dataset.csv",index = False)
            e2 = cv2.getTickCount()

            # calculate streaming metrics
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print('Streaming duration:', time0)

            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)
            
        finally:
            self.connection.close()
            self.server_socket.close()
        pygame.quit()
        quit()

if __name__ == '__main__':
    StreamingServer()