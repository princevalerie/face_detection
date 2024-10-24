import streamlit as st
import cv2
import numpy as np
from PIL import Image
import platform
import os

def check_available_cameras():
    """Check available camera devices"""
    available_cameras = []
    for i in range(5):  # Check first 5 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    st.title("Face Tracking Application")
    
    # System info display
    st.sidebar.header("System Information")
    st.sidebar.write(f"OpenCV version: {cv2.__version__}")
    st.sidebar.write(f"Platform: {platform.system()}")
    
    # Check available cameras
    if 'available_cameras' not in st.session_state:
        st.session_state.available_cameras = check_available_cameras()
    
    if not st.session_state.available_cameras:
        st.error("No cameras detected! Please check your camera connections and permissions.")
        st.info("""
        Troubleshooting tips:
        1. Pastikan browser Anda mengizinkan akses kamera
        2. Jika menggunakan Streamlit Cloud:
           - Gunakan https://
           - Izinkan akses kamera di browser
        3. Jika di local:
           - Restart aplikasi
           - Periksa apakah kamera sedang digunakan aplikasi lain
        """)
        return
    
    # Camera selection
    st.sidebar.header("Camera Settings")
    camera_index = st.sidebar.selectbox(
        "Select Camera",
        options=st.session_state.available_cameras,
        format_func=lambda x: f"Camera {x}"
    )
    
    # Parameter settings
    st.sidebar.header("Detection Settings")
    
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
    
    flip_image = st.sidebar.checkbox(
        "Flip Image Horizontally",
        value=True
    )
    
    # Load face cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return
    
    # Camera handling
    if st.button("Start Camera"):
        st.write("Initializing camera...")
        
        try:
            # Try with different backend APIs
            for api_pref in [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW]:
                video_capture = cv2.VideoCapture(camera_index, api_pref)
                if video_capture.isOpened():
                    break
            
            if not video_capture.isOpened():
                st.error("Failed to open camera. Trying alternative method...")
                # Try setting resolution explicitly
                video_capture = cv2.VideoCapture(camera_index)
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if not video_capture.isOpened():
                    st.error("Could not open camera. Please check permissions and try again.")
                    return
            
            st.success("Camera initialized successfully!")
            stframe = st.empty()
            stop_button = st.button("Stop")
            
            while not stop_button:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                if flip_image:
                    frame = cv2.flip(frame, 1)
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_face_size, min_face_size)
                )
                
                # Draw rectangles
                for (x, y, w, h) in faces:
                    hex_color = rect_color.lstrip('#')
                    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, rect_thickness)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")
            
            video_capture.release()
            st.write("Camera stopped")
            
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.info("""
            Troubleshooting:
            1. Refresh halaman
            2. Periksa izin kamera di browser
            3. Pastikan tidak ada aplikasi lain yang menggunakan kamera
            4. Coba device kamera lain jika tersedia
            """)

if __name__ == '__main__':
    main()
