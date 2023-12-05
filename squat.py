import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import cv2

st.set_page_config(layout="wide")
mp_drawing = None
mp_pose = None

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def image():
    img_file = None
    img_bytes = st.sidebar.file_uploader("upload an image", type=['png', 'jpeg', 'jpg'])

    if img_bytes:
        img_file = "data/image." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        st.markdown("---")
        output = st.empty()

        img = cv2.imread(img_file)
        output.image(process_image(img))

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500, 750))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        result = pose.process(img)

        if result.pose_landmarks is not None:
            landmarks = result.pose_landmarks.landmark

            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            angle = (r_knee_angle + l_knee_angle) / 2

            if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.3 or landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.3:
                cv2.putText(img, f"Squat Angle: {str(angle)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

                if angle <= 90:
                    cv2.putText(img, f"SQUAT GOOD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
                elif angle > 90:
                    cv2.putText(img, f"SQUAT BAD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    return img

def main():
    global mp_drawing, mp_pose

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    st.title("squat validation")

    image()

if __name__ == "__main__":
    main()
