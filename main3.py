import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

def force_camera_access(camera_index=0):
    """Force camera access with multiple attempts and configurations"""
    # List of possible backend preferences
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_MSMF, cv2.CAP_GSTREAMER]
    
    # List of possible resolutions to try
    resolutions = [
        (640, 480),
        (320, 240),
        (800, 600),
        (1280, 720)
    ]
    
    for backend in backends:
        st.write(f"Trying camera backend: {backend}")
        cap = cv2.VideoCapture(camera_index, backend)
        
        if not cap.isOpened():
            # Try setting different resolutions
            for width, height in resolutions:
                cap = cv2.VideoCapture(camera_index, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                if cap.isOpened():
                    st.success(f"Camera opened successfully with resolution {width}x{height}")
                    return cap
        else:
            st.success("Camera opened successfully with default settings")
            return cap
            
    # If all attempts fail, try one last time with default settings
    return cv2.VideoCapture(camera_index)

def main():
    st.title("Face Tracking Application (Force Access)")
    
    # System info
    st.sidebar.write(f"OpenCV version: {cv2.__version__}")
    
    # Camera selection
    camera_index = st.sidebar.number_input("Camera Index", 0, 10, 0)
    
    # Detection settings
    scale_factor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.2, 0.1)
    min_neighbors = st.sidebar.slider("Minimum Neighbors", 1, 10, 5)
    min_face_size = st.sidebar.slider("Minimum Face Size", 10, 100, 30)
    rect_color = st.sidebar.color_picker("Rectangle Color", "#00FF00")
    rect_thickness = st.sidebar.slider("Rectangle Thickness", 1, 5, 2)
    flip_image = st.sidebar.checkbox("Flip Image", True)

    # Force buffer clearing
    if 'frame_buffer' not in st.session_state:
        st.session_state.frame_buffer = None

    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if st.button("Start Camera (Force Access)"):
            st.write("Attempting to force camera access...")
            
            try:
                # Force camera access
                video_capture = force_camera_access(camera_index)
                
                if not video_capture.isOpened():
                    st.error("Failed to force camera access")
                    st.info("""
                    Mencoba solusi alternatif:
                    1. Ganti Camera Index di sidebar
                    2. Refresh browser dan izinkan akses kamera
                    3. Coba browser lain (Chrome/Firefox)
                    4. Restart komputer jika perlu
                    """)
                    return
                
                # Force some camera properties
                video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                video_capture.set(cv2.CAP_PROP_FPS, 30)
                video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
                stframe = st.empty()
                stop_button = st.button("Stop Camera")
                
                # Clear buffer
                for _ in range(5):
                    video_capture.read()
                
                while not stop_button:
                    # Read multiple frames to clear buffer
                    for _ in range(2):
                        ret, frame = video_capture.read()
                    
                    if not ret:
                        st.error("Frame capture failed. Retrying...")
                        time.sleep(0.1)
                        continue
                    
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
                    stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Add small delay to reduce CPU usage
                    time.sleep(0.01)
                
                video_capture.release()
                st.session_state.frame_buffer = None
                st.write("Camera stopped")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.write("""
                Jika masih error, coba:
                1. Ubah Camera Index (0, 1, 2, dst)
                2. Pastikan kamera tidak digunakan aplikasi lain
                3. Periksa Device Manager/System Preferences
                4. Install ulang driver kamera
                """)
                
    except Exception as e:
        st.error(f"Setup error: {str(e)}")

if __name__ == '__main__':
    main()
