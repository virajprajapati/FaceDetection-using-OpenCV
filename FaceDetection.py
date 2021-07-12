import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
while True:
ret,frame = video.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
for (x,y,w,h) in faces:
roi_gray = gray[y:y+h,x:x+w]
color = (255,0,0)
stroke = 1
width = x + w
height = y + h
cv2.rectangle(frame,(x,y),(width,height),color,stroke)
cv2.imshow("Captured image",frame)
if cv2.waitKey(1) & 0xFF == ord("q"):
break
video.release()
cv2.destroyAllWindows()