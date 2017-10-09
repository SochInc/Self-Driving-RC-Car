from flask import Flask, jsonify, request, Response
import RPi.GPIO as GPIO
import time
import json

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(2,GPIO.OUT)
GPIO.setup(3,GPIO.OUT)
GPIO.setup(4,GPIO.OUT)
GPIO.setup(17,GPIO.OUT)


GPIO.output(2,GPIO.LOW)
GPIO.output(3,GPIO.LOW)
GPIO.output(4,GPIO.LOW)
GPIO.output(17,GPIO.LOW)


app = Flask(__name__)
@app.route('/messages', methods = ['POST'])
def api_message():
        data = request.args.get('data')
        control(data)

def control(key):
        print(key)
        if key == 's': 
                print ("Backward")
                GPIO.output(2,GPIO.LOW)
                GPIO.output(3,GPIO.HIGH)
                GPIO.output(4,GPIO.HIGH)
                GPIO.output(17,GPIO.LOW)
                
        elif key == 'w':
                print ("forward")
                GPIO.output(2,GPIO.HIGH)
                GPIO.output(3,GPIO.LOW)
                GPIO.output(4,GPIO.LOW)
                GPIO.output(17,GPIO.HIGH)
        elif key == 'a':
                print ("Left")
                GPIO.output(2,GPIO.HIGH)
                GPIO.output(3,GPIO.LOW)
                GPIO.output(4,GPIO.HIGH)
                GPIO.output(17,GPIO.LOW)
        elif key == 'd':
                print ("Right")
                GPIO.output(2,GPIO.HIGH)
                GPIO.output(3,GPIO.LOW)
                GPIO.output(4,GPIO.HIGH)
                GPIO.output(17,GPIO.HIGH)
        elif key == 'e':
                print ("Forward Left")
                GPIO.output(2,GPIO.LOW)
                GPIO.output(3,GPIO.LOW)
                GPIO.output(4,GPIO.HIGH)
                GPIO.output(17,GPIO.LOW)
        elif key == 'q':
                print ("Forward Right")
                GPIO.output(2,GPIO.HIGH)
                GPIO.output(3,GPIO.HIGH)
                GPIO.output(4,GPIO.LOW)
                GPIO.output(17,GPIO.HIGH)
        elif key == 'x':
                GPIO.output(2,GPIO.LOW)
                GPIO.output(3,GPIO.LOW)
                GPIO.output(4,GPIO.LOW)
                GPIO.output(17,GPIO.LOW)
        elif key == 'quit':
                quit()

if __name__ == '__main__':
    app.run(host='192.168.1.106', port=5000, debug=False)
