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

# function to extract haralick textures from an image
def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	return ht_mean

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
	

def crop_coordenada(cur_label, image):
	
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
						print ("Processing Image - {} in {}".format(i, cur_label))
						# read the training image
						image= cv2.imread(file)
						print(file)
						# convert the image to grayscale
						atual = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

						# extract features from the image
						
						cv2.imwrite("dataset/"+cur_label+str(i)+".jpg", atual)
						try:
								cont+=1
								if cont > 9:
									anterior= cv2.imread("dataset/"+cur_label+str(i-10)+".jpg",0)
									desenhar_circulo(image, anterior, atual, name, cur_label, cont)
									
									os.remove("dataset/"+cur_label+str(i-10)+".jpg")
						except:
								print()
						# append the feature vector and label
						#features.append(features)
						#labels.append(cur_label)

						# show loop update
						i += 1



def desenhar_circulo(original, anterior, atual, name, cur_label, numero):
	frame= anterior.copy()
	ROI= atual.copy()

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
	resultado = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

	# Convert to HSV color space
	#hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	# Create a binary image with where white will be skin colors and rest is black
	#thresh = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([20, 255, 255]))

	#_, thresh = cv2.threshold(fgmask, 75, 255, cv2.THRESH_BINARY);
	# Find contours of the filtered frame
	_, contours, hierarchy = cv2.findContours(resultado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# print(contours)
	
	if contours:
			cnt = max(contours, key = cv2.contourArea)
			cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

			# Draw center mass
			#cv2.circle(ROI, center, 15, [0, 0, 255], 2)

			# find the circle which completely covers the object with minimum area
			(x, y), radius = cv2.minEnclosingCircle(cnt) #VAI
			center = (int(x), int(y))
			radius = int(radius)
			cv2.circle(ROI, center, radius, (0, 0, 0), 3)
			
			#crop around de circle############################3
			x,y,w,h= crop_coordenada(cur_label, ROI)
			output = original[y:y+h,x:x+w]
			output = cv2.resize(output, (224, 224))

			area_of_circle = math.pi * radius * radius #VAI
			
			imagem= cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
			imagem2= output.copy()
			imgYCC = cv2.cvtColor(imagem2, cv2.COLOR_BGR2YCR_CB)
			
			#Tratamento da imagem para a segmentação
			imgTratada=	cv2.GaussianBlur(imagem,(7,7),25)
			imgTratada=	cv2.bilateralFilter(imgTratada,5,105, 105)

			elementoEstruturante=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3) ) 

			#
			tipo= cv2.THRESH_BINARY_INV	+ cv2.THRESH_OTSU
			limiar,	imgBinarizada= cv2.threshold(imgTratada,0,	255,tipo)
			print(limiar)

			res= cv2.bitwise_and(imgYCC, imgYCC, mask=imgBinarizada)

			lower_ycc = np.array([0, 95-55, 95-55])
			upper_ycc = np.array([255, 95+95, 95+95])

			#lower_ycc = np.array([80,77,round(valorMedio[2])-30])
			#upper_ycc = np.array([255,190,200])
			mask_ycc= cv2.inRange(res, lower_ycc, upper_ycc)
			elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
			mask_ycc= cv2.morphologyEx(mask_ycc, cv2.MORPH_OPEN, elementoEstruturante)
			mask_ycc=	cv2.medianBlur(mask_ycc,7)
			lementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
			imagemProcessada= cv2.dilate(mask_ycc, elementoEstruturante, iterations= 2)
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

			res= cv2.bitwise_and(imagem2, imagem2, mask=imagemProcessada)

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

			lementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
			imagemProcessada= cv2.dilate(imagemProcessada, elementoEstruturante, iterations= 1)
			elementoEstruturante=cv2.getStructuringElement(	cv2.MORPH_ELLIPSE,(11,11) ) 
			imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
			
			#cv2.imwrite('dataset/final/'+cur_label+str(numero)+'.jpg', imagemProcessada)
			


			vetor= []
			vetor.append(name)
			vetor.append(cur_label+str(numero))
			#VAI
			_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# print(contours)

			if contours:
				cnta = max(contours, key = cv2.contourArea)

				cnt= cnta
				tamanho= len(cnt)
				ar = cv2.contourArea(cnt)

				# Find moments of the contour
				moments = cv2.moments(cnt)

				cx = 0
				cy = 0
				# Central mass of first order moments
				if moments['m00'] != 0:
					cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
					cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

				center = (cx, cy) #VAI

				vetor.append(cx)
				vetor.append(cy)



				print(center)
				cv2.circle(imagemProcessada, center, 15, [0, 0, 255], 2)

				#VAI
				# find the circle which completely covers the object with minimum area
				(x, y), radius = cv2.minEnclosingCircle(cnt)
				center = (int(x), int(y))
				radius = int(radius)
				cv2.circle(imagemProcessada, center, radius, (255, 0, 0), 3)

				vetor.append(x)
				vetor.append(y)
				vetor.append(radius)

				#VAI
				area_of_circle = math.pi * radius * radius

				vetor.append(area_of_circle)

				rect = cv2.minAreaRect(cnt) #VAI

				vetor.append(rect)

				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(imagemProcessada, [box], 0, (255, 0, 255), 2)



				##VAI APARTIR DAQUI
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
						cv2.line(imagemProcessada, start, end, [255, 255, 0], 3)
						cv2.circle(imagemProcessada, far, 8, [255, 84, 0], -1)
						#  finger count
						a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
						b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
						c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
						angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI

						vetor.append(angle)

						area = cv2.contourArea(cnt) #VAI #TA MAIS DE DUAS

						vetor.append(area)

						if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers #VAI ###TA MAIS DE DUAS
							fingers += 1

							vetor.append(fingers)

							cv2.circle(imagemProcessada, far, 1, [255, 0, 0], -1)

						if len(cnt) >= 5:
							(x_centre, y_centre), (minor_axis, major_axis), angle_t = cv2.fitEllipse(cnt) #VAI TA MAIS DE DUAS
							vetor.append(x_centre)
							vetor.append(y_centre)
							vetor.append(minor_axis)
							vetor.append(major_axis)
							vetor.append(angle_t)

						letter = ''
						vetor.append(area_of_circle - area) #VAI #TA MAIS DE DUAS

			cv2.imshow("b", imagemProcessada)
			cv2.waitKey(0)
			#escrever_dados("file.csv",[vetor])
train()