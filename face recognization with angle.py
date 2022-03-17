import cv2
import numpy as np
import glob
import os
import math

import matplotlib.pyplot as plt
import argparse
from scipy import ndimage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#for i in os.listdir('C:\\Users\\vamshi\\Desktop\\new'):

for j in os.listdir('E:\\New folder\\new\\21-12-2021'):

        image=cv2.imread('E:\\New folder\\new\\21-12-2021'+'//'+j)
        print(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(gray)

        angles=[]
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        print(faces)
        #half = cv2.resize(image, (100, 100), fx=0.3, fy=0.3)
        #print(half)
        for (x,y,w,h) in faces :
                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                 cv2.rectangle(image, (x, y), (x + 2 * w, y + 2 * h), (0, 255, 0), 2)
                 z = ((x + w) - (x + 2 * w))
                 print(z)
                 print((x, y, w, h))

                 for (x1, y1, x2, y2) in faces:
                         angle = math.degrees(math.atan2(y1 - x2, x1 -y2 ))
                         angles.append(angle)
                         #print(angle)
                         median_angle = np.median(angles)
                         print(f"Angle is {median_angle:.04f}")
                         cv2.namedWindow('result image', cv2.WINDOW_NORMAL)
                         cv2.imshow('result image', image)
                         cv2.waitKey(0)
                         cv2.destroyAllWindows()

















