import cv2
import numpy as np

def generarCentros():
    contornos,_ = cv2.findContours(maskConjunto,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if area > 1000:
            M = cv2.moments(contorno)
            if (M["m00"]==0): 
                M["m00"]=1
            x = int(M["m10"]/M["m00"])
            y = int(M['m01']/M['m00'])
            cv2.circle(imagen,(x,y),7,(250,250,250),-1)
            cv2.putText(imagen,'{},{}'.format(x,y),(x+10,y), font, 0.75,(0,0,0),1,cv2.LINE_AA)

def generarContorno(mascara,color):
    contornos,_ = cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contorno in contornos:
        nuevoContorno = cv2.convexHull(contorno)
        cv2.drawContours(imagen, [nuevoContorno], 0, color, 3)

imagen= cv2.imread("figurasColores.jpg")
imagenHSV = cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)

#Definimos rango de colores

rojoBajo1 = np.array([0, 100, 20])
rojoAlto1 = np.array([4, 255, 255])
rojoBajo2 = np.array([175, 100, 20])
rojoAlto2 = np.array([180, 255, 255])

azulBajo= np.array([100,50,50])
azulAlto = np.array([125, 255, 255])

verdeBajo = np.array([45,100,20])
verdeAlto = np.array([65,255,255])

amarrilloBajo = np.array([25,100,20])
amarrilloAlto = np.array([35,255,255])

#Generamos mascara binaria

maskAmarrillo = cv2.inRange(imagenHSV,amarrilloBajo,amarrilloAlto)

maskVerde = cv2.inRange(imagenHSV,verdeBajo,verdeAlto)

maskRojo1 = cv2.inRange(imagenHSV, rojoBajo1, rojoAlto1)
maskRojo2 = cv2.inRange(imagenHSV, rojoBajo2, rojoAlto2)

maskRojo =  cv2.add(maskRojo1, maskRojo2)

maskAzul = cv2.inRange(imagenHSV,azulBajo,azulAlto)

maskUnion1 = cv2.add(maskAzul,maskRojo)
maskUnion2 = cv2.add(maskUnion1,maskVerde)
maskConjunto = cv2.add(maskUnion2,maskAmarrillo)

#Generamos centros y bordes

generarCentros()
generarContorno(maskAzul,(255,0,0))
generarContorno(maskRojo,(0,0,255))
generarContorno(maskVerde,(0,255,0))
generarContorno(maskAmarrillo,(0,255,255))

cv2.imshow("Original",imagen)
cv2.imshow("maskBinario",maskConjunto)
cv2.waitKey(0)