# http://www.sethbenton.com/projects/

import numpy as np
import cv2
import time

# source = aviread('..\test_video\san_fran_traffic_30sec_QVGA_Cinepak');

path = 'persona_fondo_azul.wmv'
path2 = 'gata.wmv'
cap = cv2.VideoCapture(0)

# Inicializo variables
thresh = 25    # Umbral defindo
compNum = 5    # Indica cuanto se añade o se resta cada frame
# Mayor -> actualizacion de background rapida -> mucho movimiento
# Menor -> actualizacion de background lenta -> poco movimiento

# Leemos el primer frame
ret,frame = cap.read()

# Si se ha leido bien, iniciamos el programa
if ret == True:
    # Ponemos como backgorund el primer frame
    backgroundGris = frame.copy()
    # Convertimos el background a escala de grises
    backgroundGris = cv2.cvtColor(backgroundGris, cv2.COLOR_BGR2GRAY)


    # ------------------- Configuramos las variales de tamaño del frame -----------------------
    fr_size = backgroundGris.shape
    height, width = fr_size[:2]
    # El foreground tendra el mismo tamaño que el background
    foreground = np.zeros((height, width), np.uint8)
    # --------------------- process frames -----------------------------------
    while(1):
        # Leemos el frame
        ret, frame = cap.read()

        # Si hemos llegado al final del vídeo salimos del bucle
        if ret == False:
            break
        
        # Convrtimos el frame a escala de grises
        frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Restamos el frame actual al background
        #diff = abs(double(frameGris) - double(backgroundGris))  # cast operands as double to avoid negative overflow
        diff = cv2.absdiff(frameGris, backgroundGris)
        
        # Si la diferencia entre el frame actual y el background es mayor que el umbral,
        # el pixel esta en el foreground
        _, foreground = cv2.threshold(diff,thresh,50,cv2.THRESH_TOZERO)
        
        # Creamos una mascara con la posicion de los pixeles donde el frame es mayor que el background
        maskCompG = cv2.compare(frameGris,backgroundGris,cv2.CMP_GT)
        # frameGris > backgroundGris?
        # si -> 255
        # no -> 0
        maskCompL = cv2.compare(frameGris,backgroundGris,cv2.CMP_LT)
        # frameGris < backgroundGris?
        
        # Si el pixel del frame es mayor que el background se suma 1 al background
        cv2.add(backgroundGris, compNum, backgroundGris, maskCompG) # (No se produce desbordamiento)
        # Si el pixel del frame es menor que el background se resta 1 al background
        cv2.subtract(backgroundGris, compNum, backgroundGris, mask = maskCompL) # (No se produce desbordamiento)
        

        # movie2avi(M,'approximate_median_background','fps',30);           % save movie as avi
                       
        # Mostramos las imágenes de la cámara, el umbral y la resta
        cv2.imshow("frame", frame)
        cv2.imshow("frameGris", frameGris)
        cv2.imshow("backgroundGris", backgroundGris)
        cv2.imshow("foreground", foreground)
        
        # Capturamos una tecla para salir
        k = cv2.waitKey(20) & 0xff

        # Tiempo de espera para que se vea bien
        #time.sleep(0.015)
                           
        # Si ha pulsado la letra esc, salimos del bucle
        if k == 27:
            break

# Liberamos la cámara y cerramos todas las ventanas
cv2.destroyAllWindows()
cap.release()
