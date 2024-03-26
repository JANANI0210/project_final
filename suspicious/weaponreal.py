import cv2
import numpy as np
import pygame
import tkinter as tk
from tkinter import ttk
from twilio.rest import Client  # Import the Twilio client

# Initialize pygame mixer for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav.mp3")  # Replace with your alarm sound file

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize video capture from the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index

# Flag to track if the alarm sound is playing
alarm_playing = False

# Replace with your Twilio Account SID and Auth Token
TWILIO_ACCOUNT_SID = "AC5c1662cc2026372ec262438fe1a50cb7"
TWILIO_AUTH_TOKEN = "0f169811eea398460306b4d3aeb4f463"

# Initialize the Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to start object detection
def start_detection():
    global alarm_playing
    alarm_playing = False

    while True:
        ret, img = cap.read()

        if not ret:
            break

        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        weapon_detected = False  # Initialize a flag for weapon detection

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
                    weapon_detected = True  # Set the flag to True if a weapon is detected

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        if weapon_detected:
            print("Weapon detected!")
            if not alarm_playing:
                pygame.mixer.Sound.play(alarm_sound)  # Play the alarm sound when a weapon is detected
                alarm_playing = True
                send_twilio_notification()  # Send a Twilio notification
        else:
            print("Weapon not detected!")
            if alarm_playing:
                pygame.mixer.Sound.stop(alarm_sound)  # Stop the alarm sound if it's playing
                alarm_playing = False

        cv2.imshow("Camera Feed", img)
        key = cv2.waitKey(1)

        if key == ord('q') or key == 27:
            break

# Function to stop object detection
def stop_detection():
    global alarm_playing
    alarm_playing = False
    pygame.mixer.Sound.stop(alarm_sound)  # Stop the alarm sound

# Function to send a notification via Twilio
def send_twilio_notification():
    # Replace with your Twilio phone number and the destination phone number
    from_phone_number = "+13345181732"
    to_phone_number = "+919524368213"

    message = twilio_client.messages.create(
        body="Weapon detected! Please check the video feed.",
        from_=from_phone_number,
        to=to_phone_number
    )

    print(f"Twilio Message SID: {message.sid}")

# Create a Tkinter window
root = tk.Tk()
root.title("Object Detection App")

# Start button to initiate object detection
start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.pack()

# Stop button to stop object detection
stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack()

# Run the Tkinter main loop
root.mainloop()

# Release the camera and close OpenCV windows when the application is closed
cap.release()
cv2.destroyAllWindows()
