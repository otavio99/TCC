import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
import time
from _datetime import datetime
import math

# function to extract haralick textures from an image
def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	return ht_mean

def train():
	# load the training dataset
	path  = "dataset/train"
	nameClasse = os.listdir(path)

	# empty list to hold feature vectors and train labels
	features = []
	labels   = []

	# loop over the training dataset
	print ("[STATUS] Started extracting haralick textures..")
	for name in nameClasse:
		cur_path = path + "/" + name
		cur_label = name
		i = 1

		for file in glob.glob(cur_path + "/*.jpg"):
			print ("Processing Image - {} in {}".format(i, cur_label))
			# read the training image
			image = cv2.imread(file)

			# convert the image to grayscale
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# extract features from the image
			

			# append the feature vector and label
			#features.append(features)
			#labels.append(cur_label)

			# show loop update
			i += 1


def features():
   
	# Capture from webcam
    fgbg = cv2.createBackgroundSubtractorMOG2()
    frame= cv2.imread("C1117.jpg",0)

    # Show ROI rectangle
    #cv2.rectangle(frame, (20, 20), (300, 300), (255, 255, 2), 4)  # outer most rectangle
    ROI = cv2.imread("C1107.jpg",0)    # MOG2 Background Subtraction
    image = cv2.imread("C1107.jpg")
    fgmask = cv2.subtract(frame,ROI)
    threshold= 7
    fgmask[fgmask>=threshold]= 255
    fgmask[fgmask<threshold]= 0

    # Noise remove
    kernel = np.ones((5, 5), np.uint8)
    c1 = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    c2 = cv2.morphologyEx(c1, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(c2, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing = cv2.blur(closing, (3, 3))
    #closing= cv2.medianBlur(closing, 5)
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Convert to HSV color space
    #hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    #thresh = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([20, 255, 255]))

    #_, thresh = cv2.threshold(fgmask, 75, 255, cv2.THRESH_BINARY);
    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)

    # Draw Contours
    cntAux=[]
    cont=0
    for cnt in contours:
        tamanho= len(cnt)
        if(tamanho>30):
            color = [222, 222, 222]  # contours color
            cv2.drawContours(ROI, [cnt], -1, color, 3)
            cntAux.append(cnt)
            cont+=1

    
    for cnta in cntAux:
    
        if contours:

            cnt = cnta
            

            tamanho= len(cnt)
            ar = cv2.contourArea(cnt)
            if(ar>500):

                # Find moments of the contour
                moments = cv2.moments(cnt)

                cx = 0
                cy = 0
                # Central mass of first order moments
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                    cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

                center = (cx, cy)

                # Draw center mass
                cv2.circle(ROI, center, 15, [0, 0, 255], 2)

                # find the circle which completely covers the object with minimum area
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(ROI, center, radius, (0, 0, 0), 3)
                area_of_circle = math.pi * radius * radius

                # drawn bounding rectangle with minimum area, also considers the rotation
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(ROI, [box], 0, (0, 0, 255), 2)

                # approximate the shape
                cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

                # Find Convex Defects
                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt, hull)

                fingers = 0

                # Get defect points and draw them in the original image
                if defects is not None:
                    # print('defects shape = ', defects.shape[0])
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        cv2.line(ROI, start, end, [0, 255, 0], 3)
                        cv2.circle(ROI, far, 8, [211, 84, 0], -1)
                        #  finger count
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        area = cv2.contourArea(cnt)

                        if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                            fingers += 1
                            cv2.circle(ROI, far, 1, [255, 0, 0], -1)

                        if len(cnt) >= 5:
                            (x_centre, y_centre), (minor_axis, major_axis), angle_t = cv2.fitEllipse(cnt)

                        letter = ''
                        if area_of_circle - area < 5000:
                            #  print('A')
                            letter = 'A'
                        elif angle_t > 120:
                            letter = 'U'
                        elif area > 120000:
                            letter = 'B'
                        elif fingers == 1:
                            if 40 < angle_t < 66:
                                # print('C')
                                letter = 'C'
                            elif 20 < angle_t < 35:
                                letter = 'L'
                            else:
                                letter = 'V'
                        elif fingers == 2:
                            if angle_t > 100:
                                letter = 'F'
                            # print('W')
                            else:
                                letter = 'W'
                        elif fingers == 3:
                            # print('4')
                            letter = '4'
                        elif fingers == 4:
                            # print('Ola!')
                            letter = 'Ola!'
                        else:
                            if 169 < angle_t < 180:
                                # print('I')
                                letter = 'I'
                            elif angle_t < 168:
                                # print('J')
                                letter = 'J'

                        # Prints the letter and the number of pointed fingers and
                        print('Fingers = '+str(fingers)+' | Letter = '+str(letter))

        else:
            # prints msg: no hand detected
            cv2.putText(frame, "No hand detected", (45, 450), font, 2, np.random.randint(0, 255, 3).tolist(), 2)

    # Show outputs images
    cv2.imshow('frame', frame)
    #cv2.imshow('blur', blur)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow('thresh', thresh)
    cv2.imshow('mog2', fgmask)
    cv2.imshow('ROI', ROI)

    cv2.imshow("tela", closing)
    # Check key pressed
    #if cv2.waitKey(100) == 27:
        #break  # ESC to quit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

train()