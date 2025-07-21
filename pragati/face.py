import cv2

#load the pre-trained haar cascode detector

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#open webcam

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    #draw rectangles
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #show the result
    cv2.imshow('Face Detection',frame)

    #press 'q to end the camera
    if cv2.waitKey(1)   & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()