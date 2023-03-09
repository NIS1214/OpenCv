#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:19:56 2023

@author: nis
"""

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from ReadImg import img

#img = cv2.imread('/Users/nis/Desktop/All/OpenCV/sample2.jpg')
plt.imshow(img[:, :, : : -1])
plt.show()


age_detected = DeepFace.analyze(img, actions = ['age'])
print(age_detected)
print("Age:", age_detected[0]['age'])



