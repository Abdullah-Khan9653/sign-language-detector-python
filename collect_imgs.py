import cv2
import os
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Create a directory to store all gesture folders
DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 🧠 Change this to the letter/number you are recording
gesture_name = input("Enter gesture label (A-Z or 1-10): ").upper()
gesture_dir = os.path.join(DATA_DIR, gesture_name)

if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

        for lm in landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # Save data
        np.save(os.path.join(gesture_dir, f'{count}.npy'), np.array(data_aux))
        count += 1

        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Collecting: {gesture_name} | Count: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Collecting images', frame)

    key = cv2.waitKey(1)
    if key == ord('q') or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()
