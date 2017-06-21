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
#cap.set(3,320)
#cap.set(4,240)

# ---------------------------- Inicializo variables -------------------------------
kernel = np.ones((5,5), np.uint8)
thresh = 75    # Umbral defindo
compNum = 4    # Indica cuanto se añade o se resta cada frame
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
    # Aplicamos suavizado para eliminar ruido
    backgroundGris = cv2.GaussianBlur(backgroundGris, (21, 21), 0)

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
        
        # Si llegamos al final del video, volvemos a empezar
        if cap.get(1) >= cap.get(7):
            cap.set(1,0)
        
        # Convrtimos el frame a escala de grises
        frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicamos suavizado para eliminar ruido
        frameGris = cv2.GaussianBlur(frameGris, (9, 9), 0)

        # Restamos el frame actual al background
        diff = cv2.absdiff(frameGris, backgroundGris)
        
        # Si la diferencia entre el frame actual y el background es mayor que el umbral,
        # el pixel esta en el foreground
        _, foreground = cv2.threshold(diff,thresh,255,cv2.THRESH_BINARY)
        
        opening = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
 
        # Copiamos el umbral para detectar los contornos
        contornosimg = closing.copy()
 
        # Buscamos contorno en la imagen
        im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
        # Recorremos todos los contornos encontrados
        for c in contornos:
            # Eliminamos los contornos más pequeños
            if cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 50000:
                continue
            # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
            (x, y, w, h) = cv2.boundingRect(c)
            # Dibujamos el rectángulo del bounds
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # --------------------- Actualizamos el background -------------------------
        # Creamos mascaras con la posicion de los pixeles donde el pixel es mayor o menor que el background
        maskCompG = cv2.compare(frameGris,backgroundGris,cv2.CMP_GT) # frameGris > backgroundGris?
        maskCompL = cv2.compare(frameGris,backgroundGris,cv2.CMP_LT) # frameGris < backgroundGris?
        
        # Si el pixel del frame es mayor que el background se suma 1 al background
        cv2.add(backgroundGris, compNum, backgroundGris, maskCompG) # (No se produce desbordamiento)
        # Si el pixel del frame es menor que el background se resta 1 al background
        cv2.subtract(backgroundGris, compNum, backgroundGris, mask = maskCompL) # (No se produce desbordamiento)
        
                       
        # Mostramos las imágenes de la cámara, el umbral y la resta
        cv2.imshow("frame", frame)
        cv2.imshow("closing", closing)
        cv2.imshow("opening", opening)
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
