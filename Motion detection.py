import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#cap.set(3,320)
#cap.set(4,240)
_,prev = cap.read()

while(True):
    _,next = cap.read()
    flow = np.array(abs(np.array(next,np.float32)-np.array(prev,np.float32)),np.uint8)
    cv2.imshow('flow',flow)
    
    prev = next
    
    cv2.imshow('next',next)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()