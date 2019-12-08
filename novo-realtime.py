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
from collections import deque

def sorted_nicely( l ):
		""" Sorts the given iterable in the way that is expected.
		Required arguments:
		l -- The iterable to be sorted.
		"""
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key = alphanum_key)


def carregar_dados(path):
	arquivo = open(path, "r")
	leitor = csv.reader(arquivo, delimiter=';')
	dias= []
	for dia in leitor:
 
 		#dado = [dia]
		dias.append(dia)

	return dias

def escrever_dados(path, dados):
	
	with open(path, 'a', newline='') as csvfile:

		writer=csv.writer(csvfile, delimiter=';')
		for i in range(len(dados)):
			writer.writerow(dados[i])
	

def crop_coordenada(image):
	
	#output = image.copy()
	gray = image
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

	return x,y,w,h
	

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



def train():
	cap = cv2.VideoCapture(0)
	cont= 0
	aux=0
	r=0
	im = deque([])
	fixo= 0
	while(1):
		cont+=1
		_, frame = cap.read()
		atual= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		im.append(frame)
		#cv2.imwrite("img/"+str(cont)+".jpg", frame)
		try:
			aux+=1
			if aux > 8:
				print(aux)
				#anterior= cv2.imread("img/"+str(cont-10)+".jpg",0)
				fixo= im.popleft()
				anterior= cv2.cvtColor(fixo, cv2.COLOR_RGB2GRAY)
				#cv2.imwrite("img/"+str(cont)+".jpg", i)
				r= desenhar(frame, anterior, atual)
				cv2.imshow("c", r)
				#cv2.imshow("c", frame)
				
				#os.remove("img/"+str(cont-10)+".jpg")
		except:
			print()

		
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break

	cv2.destroyAllWindows()
	cap.release()

def desenhar(original, anterior, atual):
	frame_anterior= anterior.copy()
	frame_atual= atual.copy()

	#encontrando a diferença do frame atual para o anterior. 
	roi = cv2.subtract(frame_atual,frame_anterior)
	#realizando um threshold para binarizar a imagem.
	threshold= 15
	roi[roi>=threshold]= 255
	roi[roi<threshold]= 0
	
	#realizando operações morfológicas para corrigir defeitos na imagem.
	kernel = np.ones((5, 5), np.uint8)
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
	roi = cv2.blur(roi, (3, 3))
	#closing= cv2.medianBlur(closing, 5)
	roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	roi= cv2.dilate(roi, elementoEstruturante, iterations= 2)


	return roi
train()