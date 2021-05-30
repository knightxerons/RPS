import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import os
import mediapipe as mp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


labels = ['Paper', 'Rock', 'Scissors']
model = keras.models.load_model('./Prediction.h5')

hand_classifier = cv2.CascadeClassifier('cascade_9th.xml')
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

def func(a):
    return np.random.randint(0, 3)

cap = cv2.VideoCapture(0)

markov_chain = np.array([np.zeros(3) for i in range(3)])

while True:
    
    _, frame = cap.read()  
    frame = cv2.flip(frame, 1)
    
    image_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res_hands = hands.process(image_RGB)
    
    imgh, imgw, imgc = frame.shape
    label_position = (100, 75)
    
    if res_hands.multi_hand_landmarks:
        
        hand_features = res_hands.multi_hand_landmarks[0]
        x1 = y2 = 1e12
        x2 = y1 = -1e12
        
        for id, lnd in enumerate(hand_features.landmark):
            x, y = int(lnd.x*imgw), int(lnd.y*imgh)
            
            x1 = min(x, x1); y1 = max(y, y1);
            x2 = max(x, x2); y2 = min(y, y2)
            
        roi = image_GRAY[y2-20:y1+20, x1-20:x2+20]
        
        try:
            if np.sum([roi]) != 0:
                roi = cv2.resize(roi, (128, 128), interpolation = cv2.INTER_AREA)
                roi = roi.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                pred = model.predict(roi)
                if(np.max(pred)>0.6):
                    cv2.rectangle(frame, (x1-20, y1+20), (x2+20, y2-20), 
                                  (255, 255, 255), 2)
                    idx = np.argmax(pred)
                    res = labels[idx]
                    cv2.putText(frame, "You: " + res, label_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand Found", label_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
        except Exception as e:
            print(e)
            
        mp_draw.draw_landmarks(frame, hand_features, mp_hands.HAND_CONNECTIONS)
    
    else:
        cv2.putText(frame, "No hand Found", label_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    
    cv2.imshow('Rock Papers Scissors', frame)
    
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break
    
cap.release()
cv2.destroyAllWindows()
