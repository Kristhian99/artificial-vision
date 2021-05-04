"""
03/15/2020
Esta prueba consiste en aplicar filtros para la elimincaion de ruido
con respecto a la deteccion de colores aunado con la deteccion de objetos con
el comando blob
Cristhian Cocoletzi Cuahutle 

"""

import cv2
import sys #libreira de comandos del sistema operativo
import numpy

print('ejmeplo de camara web, Esc para salir')

#Parametros de configuracion de la camara

anchoVideo=640
altoVideo=480
fuente=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
#Creacion de objeto
camaraUSB=cv2.VideoCapture(0)
#0 es para la camara 1, el 1 es para la segunda camara, 2 es para la siguiente.....
camaraUSB.set(3,anchoVideo)
camaraUSB.set(4,altoVideo)
kernel=numpy.ones((7,7),numpy.uint8)

#creamos un rango para detctar elcolor a buscar,[HSV]
rangoinferior=numpy.array([22,100,100],numpy.uint8)
rangoSuperior=numpy.array([38,255,255],numpy.uint8)

#Creamos parametros de deteccion
#Crear parametros para el comando blob
params = cv2.SimpleBlobDetector_Params()

# Ajuste de  thresholds
params.minThreshold = 10
params.maxThreshold = 230


# Filtor de  Area.
params.filterByArea = True
params.minArea = 100

# Filtro de Circularity
params.filterByCircularity = True
params.minCircularity = 0.06

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

#Crear objeto del comando 
ver = (cv2.__version__).split('.')

if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

	
while (True):
    validarCamara,imagenOriginal=camaraUSB.read()
    if (not validarCamara):
        print("No se pudo abrir la camara...")
        break

    #Para Mostrar la camara 
    c=cv2.resize(imagenOriginal,(640,480))
    imagenflip=cv2.flip(c,1)
    cv2.imshow('Prueba Camara Web2',imagenflip)
    tecla=cv2.waitKey(10)

    "#Para mostrar la camara PACMAN"

    #Cambiamos al formato hsb
    imaformatoHSV=cv2.cvtColor(imagenflip,cv2.COLOR_BGR2HSV)
    
    #Crear mascara de deteccion de color amarillo
    maskImagen=cv2.inRange(imaformatoHSV,rangoinferior,rangoSuperior)
    mask_AND_imagenChica=cv2.bitwise_and(imagenflip,imagenflip,mask=maskImagen)

    #filtro medio de imagen
    imagen_procesada=cv2.medianBlur(maskImagen,7)
    opening=cv2.morphologyEx(imagen_procesada,cv2.MORPH_OPEN,kernel)
    Closing=cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
    resultado=cv2.bitwise_and(imagenflip,imagenflip,mask=opening)
    resultado1=cv2.bitwise_and(imagenflip,imagenflip,mask=Closing)
    "#Comando de deteccion"
    
    #Aplicacion de comando
    keypoints = detector.detect(resultado1)

    #Dibujar centro en la imagen
    
    imagen_C = cv2.drawKeypoints(imagenflip, keypoints, numpy.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #mostrar imagenes
    #cv2.imshow('Imagen Procesada',imagen_procesada)
    #cv2.imshow('imagen masacara',maskImagen)
    #cv2.imshow('Imagen Opening',opening)
    cv2.imshow('Imagen Resultado',resultado1)
    #cv2.imshow('Deteccion de objeto',imagen_C)

    if tecla==32:
        cv2.imwrite('ImagenCamara.jpg',imagenflip)
        cv2.imwrite('prueba1.jpg',resultado1 )
       
        
    if tecla==27:
        break
camaraUSB.release()#Cerrar la camara 
cv2.destroyAllWindows()
    
