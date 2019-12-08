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
	
#Na realidade essa função encontra a coordenada para crop
#Ela basicamente preenche o contorno desenhado em volta do objeto que restou da divisão e tranforma
#essa area em um objeto proprio, entao é capturado seu contorno e estraido suas coordenadas com boundingRect.
#floodFill é utilizado para preencher o contorno.
def crop_coordenada(cur_label, image):
	
	#output = image.copy()
	gray = image.copy()
	gray2= gray.copy()
	size= gray.shape[0] * gray.shape[1]
	height,width = gray.shape
	mask = np.zeros((height,width), np.uint8)
	# detect circles in the image
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, size)
	
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
	
'''
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
'''

#essa funcao desenha um circulo na regiao de interesse
def desenhar_roi(original, anterior, atual, name, cur_label, numero):
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
	#acima fim do primeiro processo
	#Imagem com fragmentos que podem ser usados para encontrar a mão.
	#Abaixo se tenta encontrar a area da mao ao tentar desenhar um circuferencia em volta dela
	_, contours, hierarchy = cv2.findContours(resultado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	if contours:
		cnt = max(contours, key = cv2.contourArea)
		cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

		# Draw center mass
		#cv2.circle(ROI, center, 15, [0, 0, 255], 2)

		# find the circle which completely covers the object with minimum area
		(x, y), radius = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		radius = int(radius)
		cv2.circle(ROI, center, radius, (0, 0, 0), 3)
		
		#O espaço onde a mão possivelmente esta vai ser recortado da imagem original utilizando a funcao abaixo
		x,y,w,h= crop_coordenada(cur_label, ROI)
		output = original[y:y+h,x:x+w]
		#essa imagem esta cinza
		output = cv2.resize(output, (224, 224))
		
		
		imagem= cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		imagemOriginal= output.copy()
		#imgYCC = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2YCR_CB)
		
		
		#Tratamento da imagem para a binarização. Com essa binarização teremos uma mascara
		imgTratada=	cv2.GaussianBlur(imagem,(7,7),25)
		imgTratada=	cv2.bilateralFilter(imgTratada,5,105, 105)
		tipo= cv2.THRESH_BINARY_INV	+ cv2.THRESH_OTSU
		limiar,	imagemProcessada= cv2.threshold(imgTratada,0,	255,tipo)
		
		
		
		'''#uni a imagem binarizada(mascara) com a imagem em ycrcb.
		res= cv2.bitwise_and(imgYCC, imgYCC, mask=imgBinarizada)


		lower_ycc = np.array([0, 95-55, 95-55])
		upper_ycc = np.array([255, 95+95, 95+95])
		mask_ycc= cv2.inRange(res, lower_ycc, upper_ycc)
		'''

		'''#size= height*width
		_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours, key = cv2.contourArea)

		cont= 0
		for i in range(len(contours)):
			if(cv2.contourArea(cnt) > cv2.contourArea(contours[i])):
				cv2.drawContours(imagemProcessada, [contours[i]], 0, (0,0,0), cv2.FILLED)
				cont+=1
'''

		
		res_anterior= imagemProcessada.copy()


		res= cv2.bitwise_and(imagemOriginal, imagemOriginal, mask=imagemProcessada)
		
		#realiza canny detecção para ajudar a separar a mão de outros objetos
		res_canny= cv2.Canny(res, 100, 200)
		
		#operações morfologicas para deixar os contornos mais uniformes e espessos
		elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
		imgTratada= cv2.morphologyEx(res_canny, cv2.MORPH_CLOSE, elementoEstruturante)
		imgTratada= cv2.dilate(imgTratada, elementoEstruturante, iterations= 1)
		imgTratada=	cv2.medianBlur(imgTratada,7)
		e= np.ones((3, 3), np.uint8)
		imgTratada=	cv2.morphologyEx(imgTratada,	cv2.MORPH_CLOSE, e)
		imgTratada=	cv2.erode(imgTratada, e, iterations= 1)

		#binariza para inverter os pixels e tornar os contornos brancos em pretos e depois somar com o resultado anterior para separar a mão dos objetos o qual estava 'grudada'
		tipo= cv2.THRESH_BINARY_INV	+ cv2.THRESH_OTSU
		limiar,	imgBinarizada= cv2.threshold(imgTratada,0,	255,tipo)
		imagemProcessada= cv2.bitwise_and(res_anterior, res_anterior, mask=imgBinarizada)
		
		
		#supõe-se que o maior contorno na imagem seja o da mão, então todos os contornos menor que este é objetos indesejaveis
		#Para deixar somente a mao na imagem o codigo abaixo procura os objetos com contorno iferior ao da mao e os preenche com preto eliminando-os assim
		_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours, key = cv2.contourArea)
		for i in range(len(contours)):
			if(cv2.contourArea(cnt) > cv2.contourArea(contours[i])):
				cv2.drawContours(imagemProcessada, [contours[i]], 0, (0,0,0), cv2.FILLED)
				

		#operações morfologicas para eliminar o massimo de falha como buracos no objeto de interesse
		lementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
		imagemProcessada= cv2.dilate(imagemProcessada, elementoEstruturante, iterations= 1)
		elementoEstruturante=cv2.getStructuringElement(	cv2.MORPH_ELLIPSE,(11,11) ) 
		imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
		
		
		
		
		
		
		
		#acima ocorre a seg final, abaixo ocorre a extraçao de caracteristicas
		
		
		
		
		
		
		
		#cv2.imwrite('dataset/final/'+cur_label+str(numero)+'.jpg', imagemProcessada)
		vetor= []
		
		#huimg = cv2.cvtColor(imagemProcessada, cv2.COLOR_BGR2GRAY)

		# Calculate Moments of the Image
		# Calculate Hu Moments
		
		#VAI
		_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# print(contours)
		

		fingers = -1
		if contours:
			cnta = max(contours, key = cv2.contourArea)

			cnt= cnta
			#tamanho= len(cnt)
			ar = cv2.contourArea(cnt)

			# Find moments of the contour
			moments = cv2.moments(cnt)
			huMoments = cv2.HuMoments(moments).flatten()
			huMoments= -np.sign(huMoments)*np.log10(np.abs(huMoments))

			cx = 0
			cy = 0
			# Central mass of first order moments
			if moments['m00'] != 0:
				cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
				cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

			center = (cx, cy) #VAI

			vetor.append(cx)
			vetor.append(cy)
			cv2.circle(imagemProcessada, center, 15, [0, 0, 255], 2)
			


			#VAI
			# find the circle which completely covers the object with minimum area
			(x, y), radius = cv2.minEnclosingCircle(cnt)
			center = (int(x), int(y))
			radius = int(radius)
			cv2.circle(imagemProcessada, center, radius, (255, 0, 0), 3)

			vetor.append(int(x))
			vetor.append(int(y))
			vetor.append(int(radius))
			

			
			area_of_circle = math.pi * radius * radius #VAI

			vetor.append(area_of_circle)

			rect = cv2.minAreaRect(cnt) #VAI
			#vetor.append(rect)

			box = cv2.boxPoints(rect)
			box = np.int0(box)
			
			cv2.drawContours(imagemProcessada, [box], 0, (255, 0, 255), 2)

			# approximate the shape
			cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

			# Find Convex Defects
			hull = cv2.convexHull(cnt, returnPoints=False)
			defects = cv2.convexityDefects(cnt, hull)

			
			angulos= []
			areas= []
			outros=[]
			outros_outer= []
			outros2= []
			# Get defect points and draw them in the original image
			if defects is not None:
				fingers=0
				# print('defects shape = ', defects.shape[0])
				for i in range(defects.shape[0]):
					s, e, f, d = defects[i, 0]
					start = tuple(cnt[s][0])
					end = tuple(cnt[e][0])
					far = tuple(cnt[f][0])
					cv2.line(imagemProcessada, start, end, [255, 255, 0], 3)
					cv2.circle(imagemProcessada, far, 8, [255, 84, 0], -1)
					a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
					b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
					c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
					angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI

					angulos.append(angle)

					area = cv2.contourArea(cnt) #VAI

					areas.append(area)

					if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers #VAI ###TA MAIS DE DUAS
						fingers += 1
						cv2.circle(imagemProcessada, far, 1, [0, 0, 0], -1)

					if len(cnt) >= 5:
						(x_centre, y_centre), (minor_axis, major_axis), angle_t = cv2.fitEllipse(cnt) #VAI
						outros.append(x_centre)
						outros.append(y_centre)
						outros.append(minor_axis)
						outros.append(major_axis)
						outros.append(angle_t)
						outros_outer.append(outros)
						
					outros2.append(area_of_circle - area) #VAI
					
				#vetor.append(angulos)
				#vetor.append(areas)
				
				res= sum(areas)
				tamanho= len(areas)
				media= res/tamanho #Vai, media das areas
				
				#vetor.append(media)
				#vetor.append(outros_outer) linha 347
				#vetor.append(outros2)  linha 349
				vetor.append(fingers)
				
				'''vetor.append(int(huMoments[0]))
				vetor.append(int(huMoments[1]))
				vetor.append(int(huMoments[2]))
				vetor.append(int(huMoments[3]))
				vetor.append(int(huMoments[4]))
				vetor.append(int(huMoments[5]))
				vetor.append(int(huMoments[6]))'''

				
		vetor.append(name)			
		#cv2.imshow("b", imagemProcessada)
		#cv2.waitKey(0)
		if fingers>=0:
			return [vetor]	

def train():
		# load the training dataset
		path  = "dataset/train"
		nameClasse = os.listdir(path)
		
		# loop over the training dataset
		for name in nameClasse:
			cur_path=path + "/" + name
			cur_label=name
			i=1
			files=  glob.glob(cur_path + "/*.jpg")
			cont=0 #determina o numero de fotos salva em dataset para a subtração.
		
			for file in sorted_nicely(files):
				print ("Processing Image - {} in {}".format(i, cur_label))
				
				image= cv2.imread(file)
				print(file)
				
				# convert the image to grayscale
				atual = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
				cv2.imwrite("dataset/"+cur_label+str(i)+".jpg", atual)
				try:
					
					cont+=1
					#conta diz quantas imagens atras sera subtraida pela atual
					if cont > 9:
						anterior= cv2.imread("dataset/"+cur_label+str(i-10)+".jpg",0)
						#desenhar circulo desenha um circulo em volta da regiao de interesse (roi)
						#resultado contem um vetor com caracteristicas extraídas
						resultado= desenhar_roi(image, anterior, atual, name, cur_label, cont)
						if resultado is not None:
							escrever_dados("file.csv",resultado)
						else: 
							cont-=1
							
						os.remove("dataset/"+cur_label+str(i-10)+".jpg")
				except:
					print()
					
				if cont>261:
					break
				# append the feature vector and label
				#features.append(features)
				#labels.append(cur_label)

				# show loop update
				i += 1
train()