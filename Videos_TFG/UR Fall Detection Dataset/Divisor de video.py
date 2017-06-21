import numpy as np
import cv2

pathS = '..\Prog\Videos_TFG\Caida JerseyBlanco.wmv'
path = 'fall-11-cam0.mp4'
path2 = 'adl-01-cam0.mp4'
newstr = path2[0:-4] # Le quitamos '.mp4'
mod = '_MOD'

cap = cv2.VideoCapture(path2)

a = cap.isOpened()
if a:
    
    print('Ancho: ',cap.get(3))
    print('Altura: ',cap.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(newstr+mod)+'.avi',fourcc, 30.0, (320,240))


    while(a):
        ret,frame = cap.read()
    
        if not ret:
            break
        roi = frame[:,320:].view()
        
        # write the flipped frame
        out.write(roi)
        
        cv2.imshow('frame',frame)
        cv2.imshow('roi',roi)
        
        # Capturamos una tecla para salir
        key = cv2.waitKey(25) & 0xFF
        
        # Si ha pulsado la letra esc, salimos
        if key == 27:
            break
else: print('\nEl archivo no existe')
    
out.release()
cap.release()
cv2.destroyAllWindows()