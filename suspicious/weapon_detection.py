import cv2
import numpy as np
import urllib
import requests
from urllib.request import urlopen
import subprocess
import os
import glob
import smtplib
import base64
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#https://outlook.live.com/mail/0/options/mail/accounts/popImap
yahoo_user = "vigneshkumaran16261@outlook.com"
yahoo_pwd = "amma@12345A"
FROM = 'vigneshkumaran16261@outlook.com'
TO = ['vigneshkumaran16261@outlook.com'] #must be a list

ip_cam=["192.168.0.185:8080"]

def mail():
    #mail system
    msg = MIMEMultipart()
    time.sleep(1)
    msg['Subject'] ="SUSPISIOUS ACTIVITY DETECTION"

    #BODY with 2 argument
    #variable = maps_url
    #body=sys.argv[1]+sys.argv[2]
    body= "Weapon Detection Near Door "         
    msg.attach(MIMEText(body,'plain'))
    time.sleep(1)


    ###IMAGE
    fp = open("1.png", 'rb')   		
    time.sleep(1)
    img = MIMEImage(fp.read())
    time.sleep(1)
    fp.close()
    time.sleep(1)
    msg.attach(img)
    time.sleep(1)


    try:
            server = smtplib.SMTP("smtp.office365.com", 587) #or port 465 doesn't seem to work!
            print ("smtp.outlook")
            server.ehlo()
            print ("ehlo")
            server.starttls()
            print ("starttls")
            server.login(yahoo_user, yahoo_pwd)
            print ("reading mail & password")
            server.sendmail(FROM, TO, msg.as_string())
            print ("from")
            server.close()
            print ('successfully sent the mail')
    except:
            print ("failed to send mail")

# Load Yolo
# Download weight file(yolov3_training_2000.weights) from this link :- https://drive.google.com/file/d/10uJEsUpQI3EmD98iwrwzbD4e19Ps-LHZ/view?usp=sharing
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i [0]- 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading image
# img = cv2.imread("room_ser.jpg")
# img = cv2.resize(img, None, fx=0.4, fy=0.4)

# Enter file name for example "ak47.mp4" or press "Enter" to start webcam


# for video capture
#cap = cv2.VideoCapture(value())

# val = cv2.VideoCapture()
while True:
   for ip in range(len(ip_cam)):
            
        url="http://"+ip_cam[ip]+"/shot.jpg"
            
        imgPath=urllib.request.urlopen(url)
            
        imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
            
        img=cv2.imdecode(imgNp,-1) 
        #_, img = cap.read()
        height, width, channels = img.shape
        # width = 512
        # height = 512

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()
