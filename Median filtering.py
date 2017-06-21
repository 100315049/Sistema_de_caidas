import numpy as np
import cv2
import time

path = 'persona_fondo_azul.wmv'
path2 = 'gata.wmv'
cap = cv2.VideoCapture(0)

# Inicializo variables
thresh = 25    # Umbral defindo
buffSize = 7    # Tamaño del buffer
# mirar ----Mayor -> actualizacion de background rapida -> mucho movimiento
#       ----Menor -> actualizzacion de background lenta -> poco movimiento

# Inicializo el buffer
buffer = []

# Leemos el primer frame
ret,frame = cap.read()

# Si se ha leido bien, iniciamos el programa
if ret == True:
    
    # Pasamos el primer frame a escala de grises
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Añadimos el primer frame al buffer
    buffer.append(frameGris)
    # Ponemos como backgorund el primer frame
    backgroundGris = buffer[0].copy()
    
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

        # Calculamos la mediana del buffer
        bufferOrdenado = np.sort(buffer, axis = 0)  # Ordenamos elemento a elemento del buffer
        backgroundGris = bufferOrdenado[int(len(buffer)/2)] # Cogemos el frame de la posicion intermedia (mediana)
        
        # Restamos el frame actual al background
        diff = cv2.absdiff(frameGris, backgroundGris)
        
        # Si la diferencia entre el frame actual y el background es mayor que el umbral,
        # el pixel esta en el foreground
        _, foreground = cv2.threshold(diff,thresh,50,cv2.THRESH_TOZERO)

        # Añadimos los frames al buffer
        if (len(buffer)<buffSize):
            buffer.append(frameGris)
        else:
            del(buffer[0])
            buffer.append(frameGris)
                       
        # Mostramos las imágenes de la cámara, el umbral y la resta
        cv2.imshow("frame", frame)
        cv2.imshow("frameGris", frameGris)
        cv2.imshow("backgroundGris", backgroundGris)
        cv2.imshow("foreground", foreground)
        
        # Capturamos una tecla para salir
        k = cv2.waitKey(1) & 0xff

        # Tiempo de espera para que se vea bien
        #time.sleep(0.015)
                           
        # Si ha pulsado la letra esc, salimos del bucle
        if k == 27:
            break

# Liberamos la cámara y cerramos todas las ventanas
cv2.destroyAllWindows()
cap.release()