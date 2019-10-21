import cv2 as cv
import numpy as np
import glob


from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

width = 64
height = 64
chanels = 3

def load_images():

    files = []
    for filename in glob.glob("files/*.jpg"):
        image = cv.imread(filename,1)
        resized = cv.resize(image, (width,height), interpolation=cv.INTER_CUBIC)
        files.append(resized)

    images = np.asanyarray(files)

    return images

def load_labels():

    labels = np.genfromtxt('labels.txt', delimiter=" ")[:,0]

    return labels


def start_webcam():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    print("asd")
    cap.release()
    cv.destroyAllWindows()

    return

if __name__ == '__main__':
    images = load_images()
    labels = load_labels()

    train_imgs, test_imgs, train_labels, test_labels = \
        train_test_split(images, labels, test_size=0.2)

    model = Sequential()
    input_shape = images[0].shape

    #64x64x3
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",
                     input_shape=input_shape))
    #32x32x32
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

    #16x16x32
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

    #8x8x32
    model.add(MaxPool2D(2,2))

    #1x2048
    model.add(Flatten())

    #1x128
    model.add(Dense(128, activation="relu"))

    #prevent overfitting
    model.add(Dropout(0.5))

    #final layer, 1 output
    model.add(Dense(1, activation="sigmoid"))

    model.summary()