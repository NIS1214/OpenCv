#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:26:46 2023

@author: nis
"""

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from ReadImg import img

#img = cv2.imread('/Users/nis/Desktop/All/OpenCV/sample2.jpg')
plt.imshow(img[:, :, : : -1])
plt.show()


result = DeepFace.analyze(img, actions = ['gender'])
print(result)

# access the correct element in the list (assuming it's the first element)
result = result[0]

# get the highest emotion value and its label
gender_detected = max(result['gender'].items(), key=lambda x: x[1])

# print the highest emotion label
print("Gender Detected:", gender_detected[0])