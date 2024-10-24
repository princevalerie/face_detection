import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class FaceTrackingTransformer(VideoTransformerBase):
    def __init__(self, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness):
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.rect_color = tuple(int(rect_color[i:i+2], 16) for i in (1, 3, 5))  # Convert hex to RGB
        self.rect_thickness = rect_thickness

    def transform(self, frame):
        # Convert the frame to an OpenCV format
        img = frame.to_ndarray(format="bgr")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), self.rect_color, self.rect_thickness)

        return img

def main():
    st.title("Face Tracking Application")

    # Sidebar settings
    st.sidebar.header("Parameter Settings")

    # Detection parameters
    scale_factor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.2, 0.1)
    min_neighbors = st.sidebar.slider("Minimum Neighbors", 1, 10, 5)
    min_face_size = st.sidebar.slider("Minimum Face Size", 10, 100, 30)
    rect_color = st.sidebar.color_picker("Rectangle Color", "#00FF00")
    rect_thickness = st.sidebar.slider("Rectangle Thickness", 1, 5, 2)

    # Start webcam with face tracking
    webrtc_streamer(key="face-tracking", 
                     video_transformer_factory=lambda: FaceTrackingTransformer(scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness))

if __name__ == '__main__':
    main()
