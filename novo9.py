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
def desenhar_roi(original, name, cur_label, numero):
	
	original=	cv2.medianBlur(original,7)
	frente= original.copy()
	frente = cv2.cvtColor(frente, cv2.COLOR_BGR2YCR_CB)
	fundo= frente.copy()
	
	fundo= fundo-10
	diferenca= frente-fundo
	y,cr,cb= cv2.split(diferenca)
	yf,crf,cbf= cv2.split(frente)
	diferenca= y-yf
	
	#diferença
	frente_grey = cv2.cvtColor(frente.copy(), cv2.COLOR_BGR2GRAY)
	fgmask= frente_grey-diferenca
	
	mean=cv2.mean(fgmask)
	print(mean)
	#binarização
	tipo= cv2.THRESH_BINARY	+ cv2.THRESH_OTSU
	limiar,fgmask= cv2.threshold(fgmask,0,	255,tipo)
		
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
	mask= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, elementoEstruturante)
	res_antes_canny= mask.copy()
	res= cv2.bitwise_and(frente, frente, mask=mask)
	
	
	
	#Canny
	res= cv2.Canny(res, 100, 200)
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	res= cv2.dilate(res, elementoEstruturante, iterations= 2)

	#inverter e mesclar com resultado antes de canny
	tipo= cv2.THRESH_BINARY_INV
	limiar,	line= cv2.threshold(res,0,	255,tipo)
	imagemProcessada= cv2.bitwise_and(res_antes_canny, res_antes_canny, mask=line)
	
	
	#retira todos os objetos menores que a mao, os preenche com preto
	_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = max(contours, key = cv2.contourArea)
	for i in range(len(contours)):
		if(cv2.contourArea(cnt) > cv2.contourArea(contours[i])):
			cv2.drawContours(imagemProcessada, [contours[i]], 0, (0,0,0), cv2.FILLED)
	
	elementoEstruturante=cv2.getStructuringElement(	cv2.MORPH_ELLIPSE,(11,11) ) 
	imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
	#fiam da segmentacao
	
	imagem_bin= imagemProcessada.copy()
	#extracao	
	_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if contours:	
		cnt = max(contours, key = cv2.contourArea)
		c= cnt
		epsilon = 0.1*cv2.arcLength(cnt,True)
		cnt = cv2.approxPolyDP(cnt,epsilon,True)
	
		moments = cv2.moments(cnt)
		huMoments = cv2.HuMoments(moments).flatten()
		#huMoments= -np.sign(huMoments)*np.log10(np.abs(huMoments))

		centeroid_x = 0
		centeroid_y = 0
		# Central mass of first order moments
		if moments['m00'] != 0:
			centeroid_x = int(moments['m10'] / moments['m00'])  # cx = M10/M00
			centeroid_y = int(moments['m01'] / moments['m00'])  # cy = M01/M00
		
		center = (centeroid_x, centeroid_y) #VAI
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		
		
		x=(extLeft,extRight,extTop,extBot)
		y=tuple([int(sum(y) / len(y)) for y in zip(*x)])

		
		(x, y), radius = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		radius = int(radius)
		
		
		#reconhecimento da palma da mao
		image_with_contour=np.zeros(imagemProcessada.shape, np.uint8)
		dist=np.zeros((imagemProcessada.shape[0],imagemProcessada.shape[1]))
		cv2.drawContours(image_with_contour,contours,-1,(255,255,255),1)

		for ind_y in range(imagemProcessada.shape[0]):
			for ind_x in range(imagemProcessada.shape[1]):
				dist[ind_y,ind_x] = cv2.pointPolygonTest(contours[0],(ind_x,ind_y),True)

		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
		#ra= int(maxVal)
		#cv2.circle(imagemProcessada,maxLoc,int(maxVal),(0, 0, 0),2)
		cv2.circle(imagemProcessada,maxLoc,int(maxVal*3.5),(255, 255, 255),2)
		output= imagemProcessada.copy()
		
		
		
		
		_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours, key = cv2.contourArea)
		x,y,w,h = cv2.boundingRect(cnt)
		output = imagem_bin[y:y+h,x:x+w]
		
		_, contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours, key = cv2.contourArea)
		epsilon = 0.1*cv2.arcLength(cnt,True)
		cnt = cv2.approxPolyDP(cnt,epsilon,True)
		
		#Palma da mao
		image_with_contour=np.zeros(output.shape, np.uint8)
		dist=np.zeros((output.shape[0],output.shape[1]))
		cv2.drawContours(image_with_contour,contours,-1,(255,255,255),1)

		for ind_y in range(output.shape[0]):
			for ind_x in range(output.shape[1]):
				dist[ind_y,ind_x] = cv2.pointPolygonTest(contours[0],(ind_x,ind_y),True)

		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
		ra= int(maxVal)
		
		#circulo que contorna o objeto
		(x, y), radius = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		radius = int(radius)
		rb= radius
		
		#cv2.circle(output,maxLoc,int(maxVal),(0, 0, 0),2)
	
	
		
		import random as rng
		
		
		#desenhando
		drawing = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
		for i in range(len(contours)):
			color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
			cv2.drawContours(drawing, contours, i, color)
			#cv2.drawContours(drawing, hull_list, i, color)
		
		#convexidades
		hull = cv2.convexHull(contours[0],returnPoints = False)
		defects = cv2.convexityDefects(contours[0],hull)
		cnt= contours[0]
		defects_choosen= []
		pontos= []
		
		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]
			start = tuple(cnt[s][0])
			end = tuple(cnt[e][0])
			far = tuple(cnt[f][0])
			
			a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
			b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
			c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
			angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI
			
			
			#b é a altura do triangulo de angulo 90
			if b>=ra and b<=rb and angle<=math.pi/2:
			
				#cv2.line(drawing,start,end,[0,255,0],2)
				#cv2.circle(drawing,far,5,[0,0,255],-1)
				if s not in defects_choosen:
					defects_choosen.append(s)
				if e not in defects_choosen:
					defects_choosen.append(e)
				#pontos.append(start)
				#pontos.append(end)
	
		defects_next= []
		for defect in defects_choosen:
			for d in defects_choosen:
				if defect != d:
					if math.isclose(defect, d, rel_tol= 0.08, abs_tol=0.0):
						defects_next.append(defect)
		
						defects_choosen= diff_list(defects_choosen, defects_next)
		
		#calculando curvatura k
		if len(defects_choosen) > 0:
			for d in defects_choosen:
				#print(cnt[d])
				#for p in cnt:
				
				try:
					
					ponto= tuple(cnt[d][0])
					p1 = tuple(cnt[d+7])
					p2 = tuple(cnt[d-7])
					
					p1= tuple(p1[0])
					p2= tuple(p2[0])
					
					a = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
					b = math.sqrt((ponto[0] - p2[0]) ** 2 + (ponto[1] - p2[1]) ** 2)
					c = math.sqrt((p1[0] - ponto[0]) ** 2 + (p1[1] - ponto[1]) ** 2)
					angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI
					
					'''a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
					b = math.sqrt((ponto[0] - p1[0]) ** 2 + (ponto[1] - p1[1]) ** 2)
					c = math.sqrt((p2[0] - ponto[0]) ** 2 + (p2[1] - ponto[1]) ** 2)
					angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI
					'''
		
					
					if angle<=math.pi:
						cv2.line(drawing,p1,ponto,[0,255,0],2)
						cv2.line(drawing,p2,ponto,[0,255,0],2)
						cv2.circle(drawing,ponto,5,[0,0,255],-1)
						#print("dedo")
				
				except:
					print()
		
		else:
			distancia= 0
			for i in range(len(cnt)):
				#print(cnt[d])
				#for p in cnt:
				
				try:
					if distancia == 0:	
						ponto= tuple(cnt[i][0])
						p1 = tuple(cnt[i+7])
						p2 = tuple(cnt[i-7])
						
						p1= tuple(p1[0])
						p2= tuple(p2[0])
						
						a = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
						b = math.sqrt((ponto[0] - p2[0]) ** 2 + (ponto[1] - p2[1]) ** 2)
						c = math.sqrt((p1[0] - ponto[0]) ** 2 + (p1[1] - ponto[1]) ** 2)
						angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI
						
						'''a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
						b = math.sqrt((ponto[0] - p1[0]) ** 2 + (ponto[1] - p1[1]) ** 2)
						c = math.sqrt((p2[0] - ponto[0]) ** 2 + (p2[1] - ponto[1]) ** 2)
						angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem ### #VAI
						'''
						if angle<=math.pi/3:
							cv2.line(drawing,p1,ponto,[0,255,0],2)
							cv2.line(drawing,p2,ponto,[0,255,0],2)
							cv2.circle(drawing,ponto,5,[0,0,255],-1)
							#print("dedo")
							distancia= 5
					else:
						distancia= distancia-1
				except:
					print()
			
		#cv2.circle(drawing, center, radius, (255, 255, 255), 3)
		#cv2.circle(drawing, maxLoc, ra, (255, 255, 255), 3)
		
	cv2.imshow("dsfa",drawing)
	if cv2.waitKey(0):
		cv2.destroyAllWindows()
		
def train():
		# load the training dataset
		path  = "dataset/train"
		nameClasse = os.listdir(path)
		im = deque([])
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
				
				#im.append(image)
				try:
					
					cont+=1
					#conta diz quantas imagens atras sera subtraida pela atual
					
					#anterior= im.popleft()
					#desenhar circulo desenha um circulo em volta da regiao de interesse (roi)
					#resultado contem um vetor com caracteristicas extraídas
					desenhar_roi(image, name, cur_label, cont)
					
						
					os.remove("dataset/"+cur_label+str(i-10)+".jpg")
				except:
					print()
				
				if cont>5:
					break
				# append the feature vector and label
				#features.append(features)
				#labels.append(cur_label)

				# show loop update
				i += 1
train()