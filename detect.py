from ultralytics import YOLO
import supervision as sv
import streamlit as st
from PIL import Image
import cv2

st.set_page_config(layout="wide")
model_path = 'models/hard-hat.pt'
model = None

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=2
)

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
                st.write("can't read frames")
                st.write("exiting ....")
                break
            output.image(process_image(frame))

        cap.release()

def process_image(img):
    result = model(img, agnostic_nms=True)[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence >= 0.5]

    labels = [f"{model.model.names[class_id]}" for _, _, _, class_id, _ in detections]

    for i in range(len(labels)):
            if labels[i] == 'head':
                 labels[i] = 'no_helmet'
    
    img = box_annotator.annotate(scene=img, detections=detections, labels=labels)
    img = cv2.resize(img, (640, 480))
    return img

def main():
    global model, model_path

    st.title("pose estimation app")

    model = YOLO(model_path)

    input_option = st.sidebar.radio("select input type: ", ['image', 'video'])

    if input_option == 'image':
        image()
    else:
        video()

if __name__ == "__main__":
    main()
