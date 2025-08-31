import cv2
import winsound 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open no")
    exit()

MAX_PEOPLE = 50

while True:
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    face_count = len(faces)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Face Count: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if face_count>MAX_PEOPLE:
        cv2.putText(frame,"ALARM: Capacity Exceeded",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        winsound.Beep(1200,500)
        
    cv2.imshow("Face Detection with Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
