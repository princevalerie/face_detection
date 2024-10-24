import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

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

    # Load face cascade classifier
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return

    # Start video input
    video_input = st.video("", format="video/mp4", key="video_input")
    
    if video_input is not None:
        try:
            # Convert video input to bytes and then to PIL image
            video_bytes = video_input.read()
            image = Image.open(io.BytesIO(video_bytes))
            
            # Convert PIL image to numpy array
            frame = np.array(image)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_face_size, min_face_size)
            )

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                # Convert hex color to BGR
                hex_color = rect_color.lstrip('#')
                bgr_color = tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, rect_thickness)

            # Display the result
            st.image(frame, caption='Detected Faces', use_column_width=True)

            # Show number of faces detected
            st.write(f"Number of faces detected: {len(faces)}")

            # Show face locations
            if len(faces) > 0:
                st.write("Face locations (x, y, width, height):")
                for i, (x, y, w, h) in enumerate(faces, 1):
                    st.write(f"Face {i}: ({x}, {y}, {w}, {h})")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("""Troubleshooting:
            1. Make sure the video is clear.
            2. Ensure faces are visible in the video.
            3. Adjust detection parameters in the sidebar.
            """)

if __name__ == '__main__':
    main()
