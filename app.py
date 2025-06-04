import streamlit as st
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
import threading

# Initialize session state for sentence persistence
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False

# ---------------------- Load Trained Model ----------------------
model_path = os.path.join(os.path.dirname(__file__), 'model.p')
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# ---------------------- Mediapipe Setup ----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# ---------------------- Label Dictionary ----------------------
labels_dict = {i: chr(65 + i) for i in range(26)}

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="ISL Detector", layout="centered")
st.title("🧠 Sign Language Detector (A–Z)")
st.markdown("Detect A–Z gestures in real-time using webcam.")

# Control buttons - Always visible
col1, col2, col3, col4 = st.columns(4)

with col1:
    if not st.session_state.detection_running:
        if st.button("▶️ Start Detection", key="start"):
            st.session_state.detection_running = True
            st.rerun()
    else:
        if st.button("⛔ Stop Detection", key="stop_main"):
            st.session_state.detection_running = False
            st.rerun()

with col2:
    if st.button("⬅️ Backspace", key="backspace"):
        if st.session_state.sentence:
            st.session_state.sentence = st.session_state.sentence[:-1]

with col3:
    if st.button("␣ Space", key="space"):
        st.session_state.sentence += " "

with col4:
    if st.button("🧹 Clear All", key="clear"):
        st.session_state.sentence = ""

# Display current sentence
st.markdown("---")
if st.session_state.sentence:
    st.success(f"**Current Sentence:** {st.session_state.sentence}")
else:
    st.info("**Current Sentence:** (empty)")

st.markdown("---")

FRAME_WINDOW = st.image([])

if st.session_state.detection_running:
    cap = cv2.VideoCapture(0)
    prediction_history = deque(maxlen=15)
    stable_threshold = 10
    last_added_time = time.time()

    st.info("🔴 **Detection Active** - Show your hand gestures! Close browser tab to stop.")
    
    # Stop button
    stop_placeholder = st.empty()
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                st.error("❌ Webcam not detected!")
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

                    prediction = model.predict(np.array([data_aux]))
                    predicted_character = labels_dict[int(prediction[0])]

                    prediction_history.append(predicted_character)

                    if prediction_history.count(predicted_character) > stable_threshold:
                        if time.time() - last_added_time > 1.2:
                            st.session_state.sentence += predicted_character
                            last_added_time = time.time()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Show your hand gesture", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw sentence strip
            cv2.rectangle(frame, (0, H - 60), (W, H), (255, 255, 255), -1)
            cv2.putText(frame, st.session_state.sentence, (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Display frame
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Check for stop condition every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                # Check if detection was stopped via main button
                if not st.session_state.detection_running:
                    break
            
            # Small delay to prevent overwhelming
            time.sleep(0.03)

    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    finally:
        cap.release()
        st.success("✅ Detection stopped successfully!")

else:
    st.info("👆 Press **Start Detection** to begin gesture recognition")