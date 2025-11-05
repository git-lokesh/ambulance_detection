import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import torch
import os

# --- Configuration ---
MODEL_PATH = 'yolov8s.pt'  # Using the Medium model
CONFIDENCE_THRESHOLD = 0.6  # Requiring 60% confidence
EV_CLASSES = [5, 7] # 5='bus', 7='truck'

# --- Helper Function ---

def process_video(video_source, model):
    """
    Processes the video source, runs YOLO detection, and displays the frames.
    """
    
    is_webcam = isinstance(video_source, int)
    
    try:
        cap = cv2.VideoCapture(video_source)
    except Exception as e:
        st.error(f"Error opening video source: {e}")
        return

    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        return

    # Create Streamlit placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    while cap.isOpened():
        
        # Check for Stop button press in webcam mode
        if is_webcam and not st.session_state.run_webcam:
            st.info("Webcam feed stopped by user.")
            break
        
        ret, frame = cap.read()
        
        if not ret:
            st.write("Video processing finished or video source ended.")
            break

        # --- YOLO Detection ---
        results = model(frame, verbose=False) 

        ev_detected = False
        
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy()  
            classes = res.boxes.cls.cpu().numpy() 
            confs = res.boxes.conf.cpu().numpy()  

            for box, cls, conf in zip(boxes, classes, confs):
                
                # Check confidence *first*
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                # Check if it's an EV class
                if int(cls) in EV_CLASSES:
                    ev_detected = True
                    
                    x1, y1, x2, y2 = map(int, box) 
                    
                    label = f"AMBULANCE Detected ({conf:.2f})"
                    color = (0, 255, 0) # GREEN
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Update Dashboard Status ---
        if ev_detected:
            status_placeholder.success("✅ AMBULANCE DETECTED! ✅")
        else:
            status_placeholder.info("⚪ No Emergency Vehicle Detected. (Monitoring...)")

        # Display the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # *** THIS IS THE FIX ***
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        # *** END FIX ***

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()
    status_placeholder.empty()
    if is_webcam:
        st.session_state.run_webcam = False

# --- Main Streamlit App ---
st.set_page_config(page_title="SmartLife EV Detection", layout="wide")
st.title("🚦 SmartLife: Emergency Vehicle Detection Dashboard")
st.write(f"Using **{MODEL_PATH}** (Medium Model) & Confidence > **{CONFIDENCE_THRESHOLD}**")

# Initialize Session State for Webcam
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

# Load the YOLO model
try:
    model = YOLO(MODEL_PATH)
    st.success(f"Successfully loaded YOLO model ({MODEL_PATH}).")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()


# --- Input Selection ---
st.sidebar.title("Input Options")
input_source = st.sidebar.radio("Select input source:", 
                                ("Upload a video file", "Use Webcam"), 
                                index=0)

if input_source == "Upload a video file":
    st.session_state.run_webcam = False 
    
    uploaded_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        video_path = tfile.name 
        tfile.close() 
        
        if st.sidebar.button("Start Processing"):
            st.sidebar.info("Processing video... Please wait.")
            process_video(video_path, model) 
            os.unlink(video_path) 

elif input_source == "Use Webcam":
    st.sidebar.warning("Webcam feed will run until you press 'Stop'.")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Start Webcam"):
            st.session_state.run_webcam = True
    
    with col2:
        if st.button("Stop Webcam"):
            st.session_state.run_webcam = False

    if st.session_state.run_webcam:
        st.sidebar.info("Webcam is running... Press 'Stop' to end.")
        process_video(0, model) # '0' is default webcam
    else:
        st.sidebar.info("Webcam is stopped.")