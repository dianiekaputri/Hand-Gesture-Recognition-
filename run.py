import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

import pickle
# load the pre-trained model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


import cv2 as cv

# open the video capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define the output video parameters
output_filename = 'compression/output_video.mp4'
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fps = 30
output_codec = cv2.VideoWriter_fourcc(*'mp4v')

# Create the video writer
video_writer = cv2.VideoWriter(output_filename, output_codec, output_fps, (output_width, output_height))

# i = 0    
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # perform image processing dan classification
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    print(y_pred)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    
    # Using cv2.putText() method
    # frame = cv2.putText(frame, str(y_pred[0]), org, font, 
    #                 fontScale, color, thickness, cv2.LINE_AA)
    # cv.imshow('frame', frame)
    # if cv.waitKey(1) == ord('q'):
    #     break

    frame = cv2.putText(frame, str(y_pred[0]), org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Write the frame to the output video
    video_writer.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and writer
cap.release()
video_writer.release()
cv2.destroyAllWindows()