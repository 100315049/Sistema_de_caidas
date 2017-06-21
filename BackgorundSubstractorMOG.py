import numpy as np
import cv2

# C:\Users\sergio\Desktop\TFG\Prog\Videos_TFG
path = '..\Prog\Videos_TFG\persona_fondo_azul.wmv'
path2 = '..\Prog\Videos_TFG\Gata.wmv'
path3 = '..\Prog\Videos_TFG\Caida JerseyBlanco.wmv'
# C:\Users\sergio\Desktop\TFG\Prog\Videos_TFG\UR Fall Detection Dataset
path4 = 'fall-01-cam0_MOD.avi'
path5 = 'adl-03-cam0_MOD.avi'

cap = cv2.VideoCapture(path5)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)

# Deshabilitamos OpenCL, si no hacemos esto no funciona
cv2.ocl.setUseOpenCL(False)

while(1):
    # Leemos el siguiente frame
    ret, frame = cap.read()

    # Si hemos llegado al final del vídeo salimos
    if not ret:
        break

    # Si llegamos al final del video, volvemos a empezar
    if cap.get(1) >= cap.get(7):
        cap.set(1,0)

    # Aplicamos el algoritmo
    fgmask = fgbg.apply(frame)

    # Copiamos el umbral para detectar los contornos
    contornosimg = fgmask.copy()

    # Buscamos contorno en la imagen
    im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Recorremos todos los contornos encontrados
    for c in contornos:
        # Eliminamos los contornos más pequeños
        if cv2.contourArea(c) < 600:
            continue

        # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
        (x, y, w, h) = cv2.boundingRect(c)
        # Dibujamos el rectángulo del bounds
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostramos las capturas
    cv2.imshow('Camara',frame)
    cv2.imshow('Umbral',fgmask)
    cv2.imshow('Contornos',contornosimg)

    # Sentencias para salir, pulsa 's' y sale
    k = cv2.waitKey(25) & 0xff

    # Si ha pulsado la letra esc, salimos del bucle
    if k == 27:
        break
    if k == ord('s'):
        cv2.waitKey(0) & 0xff
 
# Liberamos la cámara y cerramos todas las ventanas
cap.release()
cv2.destroyAllWindows()
