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
import statistics



def diff_list(li1, li2): 
    return (list(set(li1) - set(li2))) 

def fill(image):
	
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
	
		cv2.imshow("floo",mask)	

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

	return im_out
	
def testeImagem():
	original= cv2.imread("A170.jpg")

	original=	cv2.medianBlur(original,7)
	frente= original.copy()
	frente = cv2.cvtColor(frente, cv2.COLOR_BGR2YCR_CB)
	fundo= frente.copy()
	
	
	cv2.imwrite("imgApresentacao/grey.jpg", cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
	
	y,cr,cb= cv2.split(frente)
	elementoEstruturante=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	fgmask=cv2.morphologyEx(y,cv2.MORPH_BLACKHAT,elementoEstruturante )
	
	fgmask	= fgmask-y
	
	cv2.imshow("immmm", fgmask)

	tipo= cv2.THRESH_BINARY	+ cv2.THRESH_OTSU
	limiar,fgmask= cv2.threshold(fgmask,0,	255,tipo)
	
	cv2.imwrite("imgApresentacao/primeira-seg.jpg",fgmask)
	
	
		
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	mask= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, elementoEstruturante)
	res_antes_canny= mask.copy()
	res= cv2.bitwise_and(frente, frente, mask=mask)
	
	cv2.imwrite("imgApresentacao/soma-and.jpg",res)
	
	#Canny
	res= cv2.Canny(res, 100, 200)
	cv2.imwrite("imgApresentacao/canny.jpg",res)
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	imgTratada= cv2.morphologyEx(res, cv2.MORPH_CLOSE, elementoEstruturante)
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	imgTratada= cv2.dilate(imgTratada, elementoEstruturante, iterations= 2)
	imgTratada=	cv2.medianBlur(imgTratada,7)
	e= np.ones((3, 3), np.uint8)
	imgTratada=	cv2.morphologyEx(imgTratada,	cv2.MORPH_CLOSE, e)
	res=cv2.erode(imgTratada, e, iterations= 1)
	cv2.imwrite("imgApresentacao/canny-tradado.jpg",res)
	
	#inverter e mesclar com resultado antes de canny
	tipo= cv2.THRESH_BINARY_INV
	limiar,	line= cv2.threshold(res,0,	255,tipo)
	cv2.imwrite("imgApresentacao2/inversao.jpg",line)
	
	#soma
	imagemProcessada= cv2.bitwise_and(res_antes_canny, res_antes_canny, mask=line)
	cv2.imwrite("imgApresentacao/separacao.jpg",imagemProcessada)
	
	#retira todos os objetos menores que a mao, os preenche com preto (assumindo que o contorno da mao tenha maior area)
	_, contours, hierarchy = cv2.findContours(imagemProcessada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = max(contours, key = cv2.contourArea)
	for i in range(len(contours)):
		if(cv2.contourArea(cnt) > cv2.contourArea(contours[i])):
			cv2.drawContours(imagemProcessada, [contours[i]], 0, (0,0,0), cv2.FILLED)
	
	cv2.imwrite("imgApresentacao/seg-area.jpg",imagemProcessada)
	
	elementoEstruturante=cv2.getStructuringElement(	cv2.MORPH_ELLIPSE,(5,5) ) 
	imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
	kernel = np.ones((3, 3), np.uint8)
	imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
	imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
	elementoEstruturante= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	imagemProcessada= cv2.dilate(imagemProcessada, elementoEstruturante, iterations= 1)
	imagemProcessada=cv2.morphologyEx(imagemProcessada,cv2.MORPH_CLOSE,elementoEstruturante )
	kernel = np.ones((3, 3), np.uint8)
	
	cv2.imwrite("imgApresentacao/apos-pos-process.jpg",imagemProcessada)
	
	cv2.imshow("Fim seg", imagemProcessada)
	cv2.imwrite("fim-seg.jpg", imagemProcessada)
	#fim da segmentacao
	

	
	imagem_bin= imagemProcessada.copy()
	#extracao	
	_, contours, hierarchy = cv2.findContours(imagem_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#o vetor abaixo vai armazenar os atributos extraidos extraidas
	vetor= []
	tem_contorno= 0
	if contours:
		tem_contorno=1
		cnt = max(contours, key = cv2.contourArea)
	
	
		moments = cv2.moments(cnt)
		
		huMoments = cv2.HuMoments(moments).flatten()
		huMoments= -np.sign(huMoments)*np.log10(np.abs(huMoments))
		
		cx = 0
		cy = 0
		# Central mass of first order moments
		if moments['m00'] != 0:
			cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
			cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
		
		center = (cx, cy)
		
		vetor.append(cx)
		vetor.append(cy)
		
		
		#reconhecimento da palma da mao
		dist=np.zeros((imagem_bin.shape[0],imagem_bin.shape[1]))
		
		for ind_y in range(imagem_bin.shape[0]):
			for ind_x in range(imagem_bin.shape[1]):
				dist[ind_y,ind_x] = cv2.pointPolygonTest(cnt,(ind_x,ind_y),True)

		
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
		ra= int(maxVal)
		centerMax= maxLoc
		areaMaxInscri = cv2.contourArea(cnt)

		#circulo que contorna o objeto completamente utilizando a menor area possivel
		cnt = max(contours, key = cv2.contourArea)
		(x, y), radius = cv2.minEnclosingCircle(cnt)
		centerMin = (int(x), int(y))
		rb = int(radius)
		areaMinEnclo = cv2.contourArea(cnt)
		
		vetor.append(centerMin[0])
		vetor.append(centerMin[1])
		vetor.append(rb)
		vetor.append(areaMinEnclo)
		
			
		import random as rng
		#criando uma mascara e desenhando o contorno da mao nela (isso porque nessa mascara da para po cor, no contorno da imagem binarizada nao da)
		drawing = np.zeros((imagem_bin.shape[0], imagem_bin.shape[1], 3), dtype=np.uint8)
		for i in range(len(contours)):
			color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
			cv2.drawContours(drawing, contours, i, color)
			#cv2.drawContours(drawing, hull_list, i, color)
		
	
		cv2.circle(drawing, centerMax, ra, (150, 20, 20), 1)
		cv2.circle(drawing, centerMax, 2, (150, 20, 20), 3)
		cv2.circle(drawing, centerMin, rb, (10, 20, 150), 3)
		cv2.circle(drawing, centerMin, 2, (10, 20, 150), 3)
		
	
		#convexidades
		hull = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hull)

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
		print(defects_choosen)
		try:
			if len(defects_choosen)>0:
				
				for defect in defects_choosen:
					for d in defects_choosen:
						if defect is not d:
							if math.isclose(defect, d, rel_tol= 0.05, abs_tol=0.0):
								#defects_next.append(defect)
								defects_choosen.remove(defect)
								
								
		except:
			print()
			
		
		print(defects_choosen)
		#calculando curvatura k
		fingers= 0
		if len(defects_choosen) > 0:
			for d in defects_choosen:
				#print(cnt[d])
				#for p in cnt:
				
				try:
					
					ponto= tuple(cnt[d][0])
					p1 = tuple(cnt[d+10])
					p2 = tuple(cnt[d-10])
					
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
						fingers= fingers+1
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
						p1 = tuple(cnt[i+10])
						p2 = tuple(cnt[i-10])
						
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
							distancia= 11
							fingers= fingers+1
					else:
						distancia= distancia-1
				except:
					print()
			
		
		
		vetor.append(fingers)
		
		vetor.append(centerMax[0])
		vetor.append(centerMax[1])
		vetor.append(ra)
		vetor.append(areaMaxInscri)
		
		vetor.append(int(huMoments[0]))
		vetor.append(int(huMoments[1]))
		vetor.append(int(huMoments[2]))
		vetor.append(int(huMoments[3]))
		vetor.append(int(huMoments[4]))
		vetor.append(int(huMoments[5]))
		vetor.append(int(huMoments[6]))
		#vetor.append(nome_classe)


		cv2.imshow("Fim analise",drawing)
		cv2.imwrite("imgApresentacao/fim-analise.jpg", drawing)
		cv2.waitKey(0)
	
testeImagem()