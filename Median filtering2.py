import numpy as np
import cv2
import time

# C:\Users\sergio\Desktop\TFG\Prog\Videos_TFG
path = '..\Prog\Videos_TFG\persona_fondo_azul.wmv'
path2 = '..\Prog\Videos_TFG\Gata.wmv'
path3 = '..\Prog\Videos_TFG\Caida JerseyBlanco.wmv'
# C:\Users\sergio\Desktop\TFG\Prog\Videos_TFG\UR Fall Detection Dataset
path4 = 'fall-01-cam0_MOD.avi'
path5 = 'adl-03-cam0_MOD.avi'

cap = cv2.VideoCapture(path5)

# ---------------------------- Inicializo variables -------------------------------
kernel = np.ones((5,5), np.uint8)
thresh = 20    # Umbral defindo
buffSize = 5    # Tamaño del buffer
# Mayor -> actualizacion de background rapida -> mucho movimiento
# Menor -> actualizacion de background lenta -> poco movimiento

# Inicializo el buffer
buffer = []

# Leemos el primer frame
ret,frame = cap.read()

# Si se ha leido bien, iniciamos el programa
if ret == True:
    
    # Pasamos el primer frame a escala de grises
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicamos suavizado para eliminar ruido
    frameGris = cv2.GaussianBlur(frameGris, (9, 9), 0)
    # Añadimos el primer frame al buffer
    buffer.append(frameGris)
    # Ponemos como backgorund el primer frame
    backgroundGris = buffer[0]
    
    # ------------------- Configuramos las variales de tamaño del frame -----------------------
    fr_size = backgroundGris.shape
    height, width = fr_size[:2]
    # El foreground tendra el mismo tamaño que el background
    foreground = np.zeros((height, width), np.uint8)
    # --------------------- process frames -----------------------------------

    n = 1
    print('n. frames: ',cap.get(7))

    while(1):
        # Leemos el frame
        ret, frame = cap.read()
        
        # Si hemos llegado al final del vídeo salimos del bucle
        if ret == False:
            break
        
        
        a = cap.get(1)
        b = cap.get(7)
        n = n+1
        if (n >= 10):
            n = 0
            print('frame actual: ',cap.get(1))
        # Si llegamos al final del video, volvemos a empezar
        if cap.get(1) >= cap.get(7):
            cap.set(1,0)
        
        # Convrtimos el frame a escala de grises
        frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicamos suavizado para eliminar ruido
        frameGris = cv2.GaussianBlur(frameGris, (5, 5), 0)

        # Calculamos la mediana del buffer
        bufferOrdenado = np.sort(buffer, axis = 0)  # Ordenamos elemento a elemento del buffer
        backgroundGris = bufferOrdenado[int(len(buffer)/2)] # Cogemos el frame de la posicion intermedia (mediana)
        
        # Restamos el frame actual al background
        diff = cv2.absdiff(frameGris, backgroundGris)
        
        # Si la diferencia entre el frame actual y el background es mayor que el umbral,
        # el pixel esta en el foreground
        _, foreground = cv2.threshold(diff,thresh,255,cv2.THRESH_BINARY)
        
        opening = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        o_c = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        c_o = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
        dilation = cv2.dilate(foreground,kernel,iterations = 3)
 
        # Copiamos el umbral para detectar los contornos
        contornosimg = dilation.copy()
 
        # Buscamos contorno en la imagen
        im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
        # Recorremos todos los contornos encontrados
        for c in contornos:
            # Eliminamos los contornos más pequeños
            if cv2.contourArea(c) < 600 or cv2.contourArea(c) > 50000:
                continue
            # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
            (x, y, w, h) = cv2.boundingRect(c)
            # Dibujamos el rectángulo del bounds
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # --------------------- Actualizamos el buffer -------------------------
        # Añadimos los frames al buffer
        if (len(buffer)<buffSize):
            buffer.append(frameGris)
        else:
            del(buffer[0])
            buffer.append(frameGris)
                       
        # Mostramos las imágenes de la cámara, el umbral y la resta
        cv2.imshow("frame", frame)
        cv2.imshow("dilation", dilation)
        cv2.imshow("opening", opening)
        cv2.imshow("closing", closing)
        cv2.imshow("o_c", o_c)
        cv2.imshow("c_o", c_o)
        cv2.imshow("backgroundGris", backgroundGris)
        cv2.imshow("foreground", foreground)
        
        # Capturamos una tecla para salir
        k = cv2.waitKey(25) & 0xff

        # Tiempo de espera para que se vea bien
        #time.sleep(0.015)
                           
        # Si ha pulsado la letra esc, salimos del bucle
        if k == 27:
            break
        if k == ord('s'):
            cv2.waitKey(0) & 0xff
      
# Si NO se ha leido bien, avisamos por terminal
else:
    print("\nNo se ha podido abrir el archivo o la cámara")

# Liberamos la cámara y cerramos todas las ventanas
cv2.destroyAllWindows()
cap.release()