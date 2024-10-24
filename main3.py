import streamlit as st
import cv2
import imageio
import numpy as np

def main():
    st.title("Face Tracking Application")

    # Sidebar settings
    st.sidebar.header("Parameter Settings")

    # Detection parameters
    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.1,
        max_value=1.5,
        value=1.2,
        step=0.1
    )

    min_neighbors = st.sidebar.slider(
        "Minimum Neighbors",
        min_value=1,
        max_value=10,
        value=5
    )

    min_face_size = st.sidebar.slider(
        "Minimum Face Size",
        min_value=10,
        max_value=100,
        value=30
    )

    rect_color = st.sidebar.color_picker(
        "Rectangle Color",
        "#00FF00"
    )

    rect_thickness = st.sidebar.slider(
        "Rectangle Thickness",
        min_value=1,
        max_value=5,
        value=2
    )

    # Load face cascade classifier
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return

    # Start webcam using imageio
    if st.button("Start Face Tracking"):
        st.write("Face tracking is running. Press 'Stop' to end.")
        try:
            video_stream = imageio.get_reader('<video0>')  # '<video0>' is the default camera
            stframe = st.empty()  # Placeholder for video frame
            stop_button = st.button("Stop")

            while not stop_button:
                # Read frame from the video stream
                frame = next(video_stream)

                # Convert to numpy array
                frame = np.array(frame)

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
                    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, rect_thickness)

                # Display the frame in Streamlit
                stframe.image(frame, channels="RGB")

                # Allow a stop condition for the loop
                stop_button = st.button("Stop")

            video_stream.close()
            st.write("Face tracking stopped.")
        except Exception as e:
            st.error(f"Error accessing the camera: {str(e)}")
            st.info("""
                Troubleshooting:
                1. Make sure the camera is not being used by another application.
                2. Restart the application if the problem persists.
                3. Check camera permissions.
            """)

if __name__ == '__main__':
    main()
