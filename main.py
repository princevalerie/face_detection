import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Face Tracking Application")
    
    # Parameter settings using sidebar
    st.sidebar.header("Parameter Settings")
    
    # Scale Factor - Controls how much the image size is reduced at each image scale
    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.1,
        max_value=1.5,
        value=1.2,
        step=0.1,
        help="Seberapa besar gambar diperkecil pada setiap skala. Nilai lebih besar = deteksi lebih cepat tapi kurang akurat"
    )
    
    # Minimum Neighbors - How many neighbors each candidate rectangle should have
    min_neighbors = st.sidebar.slider(
        "Minimum Neighbors",
        min_value=1,
        max_value=10,
        value=5,
        help="Jumlah minimum tetangga yang harus dimiliki setiap kandidat. Nilai lebih tinggi = lebih sedikit deteksi palsu"
    )
    
    # Minimum Face Size
    min_face_size = st.sidebar.slider(
        "Minimum Face Size",
        min_value=10,
        max_value=100,
        value=30,
        help="Ukuran minimum wajah yang akan dideteksi (dalam piksel)"
    )
    
    # Rectangle Color
    rect_color = st.sidebar.color_picker(
        "Rectangle Color",
        "#00FF00",
        help="Warna kotak penanda wajah"
    )
    
    # Rectangle Thickness
    rect_thickness = st.sidebar.slider(
        "Rectangle Thickness",
        min_value=1,
        max_value=5,
        value=2,
        help="Ketebalan garis kotak penanda"
    )
    
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start webcam
    if st.button("Start Camera"):
        st.write("Camera is starting... Press 'Stop' to end.")
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_button = st.button("Stop")
        
        while not stop_button:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture video")
                break
                
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
                # Convert hex color to BGR
                hex_color = rect_color.lstrip('#')
                bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, rect_thickness)
            
            # Convert BGR to RGB for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        st.write("Camera stopped")

if __name__ == '__main__':
    main()