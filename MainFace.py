#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:42:42 2023

@author: nis
"""

import cv2
import matplotlib.pyplot as plt
from ReadImg import img
from EmotionDetection import highest_emotion
from EthnicityDetection import race_detected
from GenderDetection import gender_detected
from AgeDetection import age_detected




#This line creates a CascadeClassifier object and loads a pre-trained XML classifier file called haarcascade_frontalface_default.xml. 
#This classifier is used to detect faces in images.
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#This line uses the detectMultiScale method of the CascadeClassifier object to detect faces in the image stored in the img variable. 
#The 1.1 and 4 arguments control the scale factor and minimum neighbor count used by the detection algorithm.
faces = faceCascade.detectMultiScale(img, 1.1, 4)


#This line starts a loop that iterates over the faces list and unpacks the (x, y, u, v) tuples containing the coordinates and size of each detected face.
for (x, y, u, v) in faces:
    #This line draws a green rectangle around each detected face on the img image, using the coordinates and size values extracted from the (x, y, u, v) tuples.
    cv2.rectangle(img, (x,y), (x+u, y+v), (0, 255, 0), 2)
    

#This line displays the img image with the detected faces on a plot using pyplot.
plt.imshow(img)


# Define a font to use for the text labels
font = cv2.FONT_HERSHEY_SIMPLEX

## Add text labels to the image for the detected emotions, ethnicity, gender, and age
cv2.putText(img, str(highest_emotion), (10, 30), font, 0.5, (225,0,0), 2, cv2.LINE_4)
cv2.putText(img, str(race_detected), (10, 60), font, 0.5, (225,0,0), 2, cv2.LINE_4)
cv2.putText(img, str(gender_detected), (10, 90), font, 0.5, (225,0,0), 2, cv2.LINE_4)
cv2.putText(img, str(age_detected), (10, 120), font, 0.5, (225,0,0), 2, cv2.LINE_4)

#img = cv2.resize(img, (720, 840))
# Display the image again with the added text labels
plt.imshow(img)

#Convert the color space of the image from BGR to RGB and display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#Save the image with bounding boxes and text labels to a file
cv2.imwrite("/Users/nis/Desktop/All/OpenCV/output.jpg", img)
