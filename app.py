import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

# Load the custom-trained YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/last.pt', force_reload=True)

model = load_model()

# Streamlit app layout
st.title("Drowsiness Detection System")
st.text("This app uses YOLOv5 to detect drowsiness in real time.")

# Start video capture
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()

        if not ret:
            st.warning("Failed to grab frame. Please check your webcam.")
            break

        # Resize frame
        frame_resized = cv2.resize(frame, (640, 480))

        # Run YOLO model on the frame
        results = model(frame_resized)
        results_frame = np.squeeze(results.render())

        # Convert frame to RGB for Streamlit display
        results_frame_rgb = cv2.cvtColor(results_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(results_frame_rgb)

    cap.release()

# Stop the webcam
if cap:
    cap.release()
    cv2.destroyAllWindows()
