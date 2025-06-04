import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

# ---------------------- Load Trained Model ----------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ---------------------- Start Webcam ----------------------
cap = cv2.VideoCapture(0)

# ---------------------- Initialize Mediapipe ----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# ---------------------- Labels (A-Z) ----------------------
labels_dict = {i: chr(65 + i) for i in range(26)}

# ---------------------- Variables for Sentence & Prediction Smoothing ----------------------
sentence = ""
prediction_history = deque(maxlen=15)
stable_threshold = 10
last_added_time = time.time()

print("[INFO] Press 'q' to quit, 'b' for backspace, 'c' to clear.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not captured.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            prediction_history.append(predicted_character)

            if prediction_history.count(predicted_character) > stable_threshold:
                if time.time() - last_added_time > 1.2:
                    sentence += predicted_character
                    last_added_time = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        cv2.putText(frame, "Show your hand gesture", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ----- Draw sentence on white full-width strip at bottom -----
    cv2.rectangle(frame, (0, H - 60), (W, H), (255, 255, 255), -1)
    cv2.putText(frame, sentence, (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show frame
    cv2.imshow('Real-Time ISL Detector', frame)

    # Key Press Actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b') and sentence:
        sentence = sentence[:-1]
    elif key == ord('c'):
        sentence = ""

# ---------------------- Release Resources ----------------------
cap.release()
cv2.destroyAllWindows()
