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
import re
import csv

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def train():
    # load the training dataset
    path  = "dataset/train"
    nameClasse = os.listdir(path)

    # empty list to hold feature vectors and train labels
    features = []
    labels   = []
    list_of_files= []


    cont=0
    # loop over the training dataset
    print ("[STATUS] Started extracting haralick textures..")
    for name in nameClasse:
        cur_path=path + "/" + name
        cur_label=name
        i=1
        files=  glob.glob(cur_path + "/*.jpg")

        for file in sorted_nicely(files):
            print(file)

def carregar_dados(path):
	arquivo = open(path, "r")
	leitor = csv.reader(arquivo, delimiter=';')
	dias= []
	for dia in leitor:
 
 		#dado = [dia]
		dias.append(dia)

	return dias

def escrever_dados(path, dados):
	with open(path, 'w', newline='') as csvfile:
		writer=csv.writer(csvfile, delimiter=';')
		for i in range(len(dados)):
			writer.writerow(dados[i])

def crop_coordenada():
	image= cv2.imread("A310.jpg")
	#output = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray2= gray
	size= gray.shape[0] * gray.shape[1]
	height,width = gray.shape
	mask = np.zeros((height,width), np.uint8)
	# detect circles in the image
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, size)
	crop= 0
	mascaras= []

	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			#cv2.circle(output, (x, y), r, (0, 255, 0), 3)
			cv2.circle(mask, (x, y), r, (255, 255, 255), 3)
			#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		

	# Copy the thresholded image.
	im_floodfill = mask.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = mask.shape[:2]
	mask_new = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask_new, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = mask | im_floodfill_inv	

	_, contours, hierarchy = cv2.findContours(im_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	x,y,w,h = cv2.boundingRect(contours[0])
	output = gray2[y:y+h,x:x+w]
			# show the output image

	mascaras.append([x,y,w,h])
	escrever_dados('file.csv', mascaras)
	

def recorte():
	mascaras= carregar_dados('file.csv')
	x,y,w,h= mascaras[0]
	x= int(x)
	y= int(y)
	w= int(w)
	h= int(h)
	image= cv2.imread("A3100.jpg")

	output = image[y:y+h,x:x+w]
	cv2.imshow("output", output)
	cv2.waitKey(0)


import scipy.misc

def tes():
	imagem= cv2.imread("F1.jpg",0)
	imagem2= cv2.imread("F1.jpg")
	imgYCC = cv2.cvtColor(imagem2, cv2.COLOR_BGR2YCR_CB)
	
	#Tratamento da imagem para a segmentação
	imgTratada=	cv2.GaussianBlur(imagem,(7,7),55)
	

	#
	tipo= cv2.THRESH_BINARY_INV	+ cv2.THRESH_OTSU
	limiar,	imgBinarizada= cv2.threshold(imgTratada,0,	255,tipo)
	print(limiar)

	res= cv2.bitwise_and(imgYCC, imgYCC, mask=imgBinarizada)

	lower_ycc = np.array([0, 95-50, 95-50])
	upper_ycc = np.array([255, 95+90, 95+90])

	#lower_ycc = np.array([80,77,round(valorMedio[2])-30])
	#upper_ycc = np.array([255,190,200])
	mask_ycc= cv2.inRange(res, lower_ycc, upper_ycc)
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	mask_ycc= cv2.morphologyEx(mask_ycc, cv2.MORPH_OPEN, elementoEstruturante)
	mask_ycc=	cv2.medianBlur(mask_ycc,7)
	lementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	imagemProcessada= cv2.dilate(mask_ycc, elementoEstruturante, iterations= 1)
	imagemProcessada=	cv2.medianBlur(imagemProcessada,7)

	
	
	height,width = imagemProcessada.shape
	size= height*width
	mask = np.zeros((height,width), np.uint8)
	_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = max(contours, key = cv2.contourArea)

	cont= 0
	maxarea= contours[0]
	for i in range(len(contours)):
		if(cv2.contourArea(cnt) > cv2.contourArea(contours[i])):
			cv2.drawContours(imagemProcessada, [contours[i]], 0, (0,0,0), cv2.FILLED)
			cont+=1

	res_anterior= imagemProcessada.copy()

	res= cv2.bitwise_and(imgYCC, imgYCC, mask=imagemProcessada)

	res= cv2.Canny(res, 100, 200)
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	imgTratada= cv2.morphologyEx(res, cv2.MORPH_CLOSE, elementoEstruturante)
	lementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	imgTratada= cv2.dilate(imgTratada, elementoEstruturante, iterations= 1)
	imgTratada=	cv2.GaussianBlur(imgTratada,(3,3),55)
	imgTratada=	cv2.medianBlur(imgTratada,7)
	e= np.ones((3, 3), np.uint8)
	imgTratada=	cv2.morphologyEx(imgTratada,	cv2.MORPH_CLOSE, e)
	imgTratada=	cv2.erode(imgTratada, e, iterations= 1)


	tipo= cv2.THRESH_BINARY_INV	+ cv2.THRESH_OTSU
	limiar,	imgBinarizada= cv2.threshold(imgTratada,0,	255,tipo)

	#in this part the hand is reparated from the other object up above
	imagemProcessada= cv2.bitwise_and(res_anterior, res_anterior, mask=imgBinarizada)

	height,width = imagemProcessada.shape
	size= height*width
	mask = np.zeros((height,width), np.uint8)
	_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = max(contours, key = cv2.contourArea)

	cont= 0
	maxarea= contours[0]
	for i in range(len(contours)):
		if(cv2.contourArea(cnt) > cv2.contourArea(contours[i])):
			cv2.drawContours(imagemProcessada, [contours[i]], 0, (0,0,0), cv2.FILLED)
			cont+=1

	cv2.imshow("b", imagemProcessada)
	cv2.imshow("c", res_anterior)
	cv2.imshow("d", imgBinarizada)

	#cv2.imshow("im2", imagemProcessada)
	cv2.waitKey(0)

tes()



