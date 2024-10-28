import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Face Detecting Application")

    # Initialize the camera index in session state if it doesn't exist
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0  # Default to front camera

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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Option to upload an image or use camera
    option = st.selectbox("Choose an option:", ["Upload Image", "Use Camera"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert the uploaded image to an array
            image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            # Process the image
            process_image(image, face_cascade, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness)

    elif option == "Use Camera":
        
        # Streamlit camera input
        st.write("Press the button below to start face tracking.")
        video_frame = st.camera_input("Camera", key="camera")

        if video_frame is not None:
            # Convert the image to an array
            image = cv2.imdecode(np.frombuffer(video_frame.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            # Process the image
            process_image(image, face_cascade, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness)

def process_image(image, face_cascade, scale_factor, min_neighbors, min_face_size, rect_color, rect_thickness):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        cv2.rectangle(image, (x, y), (x + w, y + h), bgr_color, rect_thickness)

    # Display the image with rectangles
    st.image(image, channels="BGR")

if __name__ == '__main__':
    main()
