
import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Transformer class for face tracking using the webcam
class FaceTrackingTransformer(VideoTransformerBase):
    def __init__(self, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = (min_face_size, min_face_size)
        self.rect_color = rect_color
        self.rect_thickness = rect_thickness
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Convert hex color to BGR
            hex_color = self.rect_color.lstrip('#')
            bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), bgr_color, self.rect_thickness)

        return img

def main():
    st.title("Face Detection Application")

    # Sidebar for adjusting face detection parameters
    st.sidebar.header("Parameter Settings")

    scale_factor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.2, 0.1)
    min_neighbors = st.sidebar.slider("Minimum Neighbors", 1, 10, 5)
    min_face_size = st.sidebar.slider("Minimum Face Size", 10, 100, 30)
    rect_color = st.sidebar.color_picker("Rectangle Color", "#00FF00")
    rect_thickness = st.sidebar.slider("Rectangle Thickness", 1, 5, 2)

    # Option to choose face detection method
    option = st.selectbox("Choose an option:", ["Upload Image", "Take a Photo", "Real-Time Face Tracking"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert the uploaded image to a numpy array
            image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            process_image(image, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness)

    elif option == "Take a Photo":
        # Take a snapshot using camera input
        snapshot = st.camera_input("Take a photo")
        if snapshot is not None:
            # Convert the snapshot image to a numpy array
            image = cv2.imdecode(np.frombuffer(snapshot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            process_image(image, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness)

    elif option == "Real-Time Face Tracking":
        st.write("Press the button below to start real-time face tracking.")

        # WebRTC streamer with the transformer for real-time video face tracking
        webrtc_streamer(
            key="face-tracking",
            video_transformer_factory=lambda: FaceTrackingTransformer(
                scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness
            )
        )

# Function to process and display the image with detected faces
def process_image(image, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        # Convert hex color to BGR
        hex_color = rect_color.lstrip('#')
        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(image, (x, y), (x + w, y + h), bgr_color, rect_thickness)

    # Display the processed image
    st.image(image, channels="BGR")

if __name__ == '__main__':
    main()
