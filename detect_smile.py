import cv2 as cv
import numpy as np

from keras.models import load_model

model = load_model('faces_model.h5')
#model = load_model('images_model.h5')

widht = 64
height = 64

def start_webcam():

    face_cascade = cv.CascadeClassifier('haar_face.xml')
    cap = cv.VideoCapture(0)
    smile = 0

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(face) > 0:
            (x, y, w, h) = face[0]
            cv.rectangle(frame, (x,y), (x+w,y+h),(0,255,0), 2)
            face_crop = frame[y:(y+h),x:(x+w)]
            #cv.imshow('crop', face_crop)
            resized_face = cv.resize(face_crop, (widht, height), interpolation=cv.INTER_CUBIC)
            resized_face = np.asanyarray([resized_face])
            smile = predict_smile(resized_face)

        if smile > 0.5:
            cv.putText(frame, "Smiling", (420,200), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255))
        else:
            cv.putText(frame, "No smile", (420, 200), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    return

def predict_smile(image):
    prediction = model.predict(image)
    return prediction[0][0]


if __name__ == '__main__':
    start_webcam()