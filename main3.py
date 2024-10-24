import streamlit as st
import cv2
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

    # Start webcam
    if st.button("Start Face Tracking"):
        st.write("Face tracking is running. Press 'Stop' to end.")
        try:
            video_capture = cv2.VideoCapture(0)  # 0 is the default camera
            stframe = st.empty()  # Placeholder for video frame
            stop_button = st.button("Stop")

            while not stop_button:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture video from the camera.")
                    break

                # Convert to grayscale for face detection
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
                    # Convert hex color to BGR
                    hex_color = rect_color.lstrip('#')
                    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, rect_thickness)

                # Convert BGR to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
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
