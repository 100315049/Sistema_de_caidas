import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8) # "Convulcionar una imagen, osea filtrar"

while(True):
    ret, frame = cap.read()
    
    # Definimos rangos de colores a detectar
    rangomax = np.array([50,255,50])
    rangomin = np.array([0,51,0])
    # Hacemos una mascara con el frame cuyos valores entren dentro de los rangos
    mascara = cv2.inRange(frame, rangomin, rangomax)
    # Eliminamos ruido
    opening = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    # Dibujamos el rectangulo y circulo
    x,y,w,h, = cv2.boundingRect(opening)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.circle(frame,(int(x+w/2),int(y+h/2)),5,(0,0,255),-1)
    
    cv2.imshow('camara',frame)
    cv2.imshow('mascara',mascara)
    
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    
cv2.destroyAllWindows()
cap.release()