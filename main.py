import keras
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import os


def func(a):
    return np.random.randint(0, 3)


labels = ['Paper', 'Rock', 'Scissors']

model = keras.models.load_model('./Prediction.h5')

hand_classifier = cv2.CascadeClassifier('cascade_second_9th.xml')

cap = cv2.VideoCapture(0)

prev = -1

while True:
    ret, frame = cap.read()
    res = []
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """
        
        roi=gray[100:350,100:350]
        cv2.rectangle(frame,(100,100),(350,350),(0,0,255),thickness=3)
        roi = cv2.resize(roi,(128,128),interpolation=cv2.INTER_AREA)
        roi = roi.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        pred = model.predict(roi)
        res = labels[np.argmax(pred)]
        label_position = (100,75)
        cv2.putText(frame,"You: "+res,label_position,cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),2)
        tp=np.argmax(pred)
        if(prev!=tp):
            print(pred)
            cp=func(tp)
            prev=tp
        c_label_position=(100,375)
        cv2.putText(frame,"Model : "+labels[cp],c_label_position,cv2.FONT_HERSHEY_SIMPLEX,1.3,(225,0,0),2)
        
        """
    hand = hand_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

    for (x, y, w, h) in hand:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
        if np.sum([roi]) != 0:
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi)
            res = labels[np.argmax(pred)]
            label_position = (100, 75)
            cv2.putText(frame, "You: " + res, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
            tp = np.argmax(pred)
            if (prev != tp):
                print(pred)
                cp = func(tp)
                prev = tp
            c_label_position = (100, 375)
            cv2.putText(frame, "Model : " + labels[cp], c_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (225, 0, 0), 2)
        else:
            cv2.putText(frame, "No Hand Detected", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (225, 0, 0), 2)

    cv2.imshow('RPS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
cv2.destroyAllWindows()
