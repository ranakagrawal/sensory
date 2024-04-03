#!/usr/bin/env python
# coding: utf-8

# # 1. Import and Install Dependencies

# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# # 2. Keypoints using MP Holistic

# In[2]:


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# In[3]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# In[4]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


# In[5]:


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# # 3. Extract Keypoint Values

# In[6]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# # 4. Setup Folders for Collection

# In[7]:


frame_height = 1088
frame_width = 1920


# In[8]:


# import cv2
# import numpy as np
# import os

# Function to process each video file as a sequence
def process_video_sequence(action_folder, video_path, sequence):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    frames = []
    
    while cap.isOpened() and frame_num < 60:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_num += 1

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)
            
        if frame_num == 1: 
            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action_folder, sequence), (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(500)
        else: 
            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action_folder, sequence), (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

        # Export keypoints
        keypoints = extract_keypoints(results)
        npy_dir = os.path.join(EXTRACTED_DATA_PATH, action_folder, str(sequence))
        os.makedirs(npy_dir, exist_ok=True)  # Ensure the directory exists or create it
        npy_path = os.path.join(npy_dir, str(frame_num - 1) + '.npy')
        np.save(npy_path, keypoints)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Padding frames if necessary to ensure exactly 60 frames
    if frame_num < 60:
        last_frame = frames[-1]
        for i in range(60 - frame_num):
            image, results = mediapipe_detection(last_frame, holistic)
            draw_styled_landmarks(image, results)
            # Show to screen
            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action_folder, sequence), (15,12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(1)
            # Export keypoints
            keypoints = extract_keypoints(results)
            npy_dir = os.path.join(EXTRACTED_DATA_PATH, action_folder, str(sequence))
            npy_path = os.path.join(npy_dir, str(frame_num + i) + '.npy')
            np.save(npy_path, keypoints)

    cap.release()
    cv2.destroyAllWindows()

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Set the path to your dataset directory
    DATA_PATH = './Electronics'
    EXTRACTED_DATA_PATH = './Extracted_Keypoints'  # New directory for extracted keypoints

    # Loop through actions
    actions = os.listdir(DATA_PATH)
    for action_folder in actions:
        # Get the path to the action folder
        action_path = os.path.join(DATA_PATH, action_folder)
        
        # Loop through video files in the action folder
        video_files = [f for f in os.listdir(action_path) if f.endswith('.MOV') or f.endswith('.mp4')] # Adjust file extensions as needed
        for idx, video_file in enumerate(video_files):
            # Process each video file as a sequence
            video_path = os.path.join(action_path, video_file)
            sequence = idx
            process_video_sequence(action_folder, video_path, sequence)


# # 6. Preprocess Data and Create Labels and Features

# In[9]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Extracted_Keypoints') 

# Actions that we try to detect
actions = np.array(['camera','laptop','radio','screen', 'television'])

# Thirty videos worth of data 
# no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 60

# Folder start
start_folder = 0


# In[10]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[11]:


label_map = {label:num for num, label in enumerate(actions)}


# In[12]:


label_map


# In[13]:


def display_directory_tree(directory):
    print(f"+ {directory}")
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = '|   ' * (level)
        print(f"{indent}|-- {os.path.basename(root)}/")
        subindent = '|   ' * (level + 1)
        for file in files:
            print(f"{subindent}|-- {file}")

# Replace 'your_directory_path' with the path of the directory you want to display
directory_path = './Extracted_Keypoints'
display_directory_tree(directory_path)


# In[14]:


sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for sequence_folder in os.listdir(action_path):
        sequence_length = len([file for file in os.listdir(os.path.join(action_path, sequence_folder)) if file.endswith('.npy')])
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(action_path, sequence_folder, "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[15]:


np.array(sequences).shape


# In[16]:


np.array(labels).shape


# In[17]:


X = np.array(sequences)


# In[18]:


X.shape


# In[19]:


y = to_categorical(labels).astype(int)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[21]:


y_test.shape


# # 7. Build and Train LSTM Neural Network

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[23]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[25]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[26]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[30]:


model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])


# In[31]:


model.summary()


# # 8. Make Predictions

# In[32]:


res = model.predict(X_test)


# In[33]:


actions[np.argmax(res[2])]


# In[34]:


actions[np.argmax(y_test[2])]


# # 9. Save Weights

# In[35]:


model.save('action.h5')


# In[36]:


# del model


# In[37]:


from keras.models import load_model

# Recreate the model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(29, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model (you may need to adjust the compilation based on your original setup)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the saved weights into the model
model.load_weights('action.h5')


# # 10. Evaluation using Confusion Matrix and Accuracy

# In[15]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[16]:


yhat = model.predict(X_test)


# In[51]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[52]:


multilabel_confusion_matrix(ytrue, yhat)


# In[53]:


accuracy_score(ytrue, yhat)


# # 11. Test in Real Time

# In[17]:


from scipy import stats


# In[18]:


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# In[19]:


plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))


# In[22]:


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture('./Adjectives/98. sick/MVI_9528.MOV')
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Sentence formation logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]
        
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# After the video ends, print the sentence formed from actions detected in the video
print(' '.join(sentence))


# In[ ]:




