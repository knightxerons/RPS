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
hands = mp_hands.Hands(static_image_mode = False,
                       max_num_hands = 1,
                       min_detection_confidence = 0.7,
                       min_tracking_confidence = 0.8)

cap = cv2.VideoCapture(0)

markov_chain = np.array([np.zeros(3) for i in range(3)])
prev = rounds = idx = 0

flag = False

def return_opp(idx):
    if(idx == 0):
        return 2
    elif(idx == 1):
        return 0
    elif(idx == 2):
        return 1

wc = wp = 0

cnt_idx = np.zeros(3)

while True:
    
    _, frame = cap.read()  
    frame = cv2.flip(frame, 1)
    
    image_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res_hands = hands.process(image_RGB)
    
    imgh, imgw, imgc = frame.shape
    label_position_player = (50, 100)
    label_position_computer = (50, 150)
    
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
                    comp_label = return_opp(np.argmax(markov_chain[prev]))
                    comp = labels[comp_label]
                    cv2.putText(frame, "Computer: " + comp, label_position_computer, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    cv2.putText(frame, "You: " + res, label_position_player, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    cnt_idx[idx] = cnt_idx[idx]+1
                    
                    flag = True
                else:
                    cv2.putText(frame, "No Hand Found", (160, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        except Exception as e:
            print(e)
            
        # mp_draw.draw_landmarks(frame, hand_features, mp_hands.HAND_CONNECTIONS)
    
    elif flag == True:
        rounds+=1
        markov_chain[prev][np.argmax(cnt_idx)]+=1
        prev = np.argmax(cnt_idx)
        flag = False
        
        if(return_opp(np.argmax(cnt_idx)) == comp_label):
            wc+=1
        elif(return_opp(comp_label) == np.argmax(cnt_idx)):
            wp+=1
            
        cnt_idx = np.zeros_like(cnt_idx)
                
    else:
        cv2.putText(frame, "No Hand Found", (160, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.putText(frame, f"Round:{rounds+1}", (200, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 2)
    
    cv2.putText(frame,"Player: {} | Computer: {}".format(wp, wc), 
                (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2) 
    
    cv2.imshow('Rock Papers Scissors', frame)
    
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break

if rounds == 0:
    rounds = 1

for i in range(3):
    for j in range(3):
        print(markov_chain[i][j], end = " ")
    print('\n')
    
print("Rounds : {}".format(rounds))

print(f"Player win Percentage: {wp/rounds*100}\nComputer win Percentage {wc/rounds*100}")

print(cnt_idx)

cap.release()
cv2.destroyAllWindows()
