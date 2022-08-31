"""
A sample program to test the model developed.

"""
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import face_recognition
from pygame import mixer

"""
  Importing all the required packages
"""
mixer.init()
#sound = mixer.Sound("")
#face_cascade = cv2.CascadeClassifier('D:\\Data\\work\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:\\Data\\work\\haarcascade_eye.xml')
faceCascade =cv2.CascadeClassifier(r'D:\\works\\detector\\haarcascade_frontalface_default.xml') 
"""
Loading opencv haarcascade to dectects eyes
"""

model = load_model(r'D:\\works\\detector\\drowsiness_detector.h5')
"""
Loading the model created with tensorflow and keras
"""


cap = cv2.VideoCapture(0)
Score = 0
while True:
    ret, frame = cap.read()
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_recognition.face_locations(frame)
    eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)       
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (h+10,x+10), (y+10,w+10), (255,255,0), 2)
    """
    Drawing a rectangle for the detected face in the camera frame using face recognition
    library
    """
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )   
        """
        Detecting the eyes and drawing bounding box on them
        """ 
        # preprocessing steps
        eye_color = frame[ey:ey+eh,ex:ex+ew]
        eye_gray = gray[ey:ey+eh,ex:ex+ew]
        eyes_new = eye_cascade.detectMultiScale(eye_gray)
        if len(eyes_new) == 0:
            print('No eyes dectected')
        else:
            for (x,y,w,h) in eyes_new:
                eyes_roi =eye_color[y:y+h,x:x+w]
                final_img = cv2.resize(eyes_roi,(224,224))
                final_img = final_img/255.0
                final_img = np.expand_dims(final_img,axis=0)
                prediction = model.predict(final_img)
                if (prediction>0):
                    status = "Open Eyes"
                else:
                    status = "Closed Eyes"

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(faceCascade.empty())
  
                for (x, y, w, h) in face: 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, status, (20,20), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                cv2.imshow('frame', frame)
  
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break       
cap.release()
cv2.destroyAllWindows()