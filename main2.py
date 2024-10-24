import streamlit as st
# Import opencv-python-headless instead of opencv-python
import cv2
import numpy as np
from PIL import Image

def main():
    # Check if opencv is installed correctly
    st.write(f"OpenCV version: {cv2.__version__}")
    
    st.title("Face Tracking Application")
    
    # Initialize session state for camera index if it doesn't exist
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0
    
    # Parameter settings using sidebar
    st.sidebar.header("Parameter Settings")
    
    # Camera Selection
    st.sidebar.header("Camera Settings")
    if st.sidebar.button("Switch Camera"):
        # Toggle between camera 0 (usually front) and 1 (usually back)
        st.session_state.camera_index = 1 if st.session_state.camera_index == 0 else 0
        st.sidebar.write(f"Current Camera: {'Front' if st.session_state.camera_index == 0 else 'Back'}")
    
    try:
        # Load the cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Scale Factor
        scale_factor = st.sidebar.slider(
            "Scale Factor",
            min_value=1.1,
            max_value=1.5,
            value=1.2,
            step=0.1
        )
        
        # Minimum Neighbors
        min_neighbors = st.sidebar.slider(
            "Minimum Neighbors",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Minimum Face Size
        min_face_size = st.sidebar.slider(
            "Minimum Face Size",
            min_value=10,
            max_value=100,
            value=30
        )
        
        # Rectangle Color
        rect_color = st.sidebar.color_picker(
            "Rectangle Color",
            "#00FF00"
        )
        
        # Rectangle Thickness
        rect_thickness = st.sidebar.slider(
            "Rectangle Thickness",
            min_value=1,
            max_value=5,
            value=2
        )
        
        # Image Flip Option
        flip_image = st.sidebar.checkbox(
            "Flip Image Horizontally",
            value=True
        )
        
        # Start webcam
        if st.button("Start Camera"):
            st.write("Camera is starting... Press 'Stop' to end.")
            try:
                video_capture = cv2.VideoCapture(st.session_state.camera_index)
                if not video_capture.isOpened():
                    st.error("Failed to open camera. Please check camera permissions.")
                    return
                
                stframe = st.empty()
                stop_button = st.button("Stop")
                
                while not stop_button:
                    ret, frame = video_capture.read()
                    if not ret:
                        st.error(f"Failed to capture video from camera {st.session_state.camera_index}")
                        break
                    
                    # Flip image if needed
                    if flip_image:
                        frame = cv2.flip(frame, 1)
                    
                    # Convert frame to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(min_face_size, min_face_size)
                    )
                    
                    # Draw rectangles around faces
                    for (x, y, w, h) in faces:
                        hex_color = rect_color.lstrip('#')
                        bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, rect_thickness)
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB")
                
                video_capture.release()
                
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                st.info("Tips:\n- Check camera permissions\n- Restart application\n- Make sure camera isn't being used by another app")
    
    except Exception as e:
        st.error(f"Setup error: {str(e)}")
        st.info("Please make sure all dependencies are installed correctly")

if __name__ == '__main__':
    main()
