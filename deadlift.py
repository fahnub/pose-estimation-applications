from landmarks import landmarks
import streamlit as st
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import cv2

st.set_page_config(layout="wide")    
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
body_lang_prob  = np.array([0,0])
body_lang_class = ''
body_language = ''
current_stage = ''
counter = 0

def video():
    vid_file = None
    vid_bytes = st.sidebar.file_uploader("upload a video", type=['mp4', 'mpv', 'avi'])
    
    if vid_bytes:
        vid_file = "data/video." + vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())

    if vid_file:
        st.markdown("---")
        output = st.empty()

        cap = cv2.VideoCapture(vid_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output.image(process_image(frame))
        cap.release()
    
    return

def process_image(img):
    global mp_drawing, mp_pose, counter_model, form_model, body_lang_prob, body_lang_class, body_language, current_stage, counter
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        result = pose.process(img)

        if result.pose_landmarks is not None:
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
            mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

            row = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten().tolist()
            x = pd.DataFrame([row], columns=landmarks)
            body_lang_prob = counter_model.predict_proba(x)[0]
            body_lang_class= counter_model.predict(x)[0]
            body_language = form_model.predict(x)[0]

            if body_lang_class == "down" and body_lang_prob[body_lang_prob.argmax()] > 0.7:
                current_stage = "down"
            elif current_stage == "down" and body_lang_class=="up" and body_lang_prob[body_lang_prob.argmax()] >0.7:
                current_stage = "up"
                counter += 1

            cv2.putText(img, f"body_language: {body_language}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
            cv2.putText(img, f"position: {current_stage}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"counter: {counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img

def main():
    global counter_model, form_model

    with open('models/deadlift.pkl', 'rb') as f:
        counter_model = pickle.load(f)
    
    with open('models/lean.pkl', 'rb') as b:
        form_model = pickle.load(b)
    
    st.title("deadlift validation app")

    video()

if __name__ == "__main__":
    main()
