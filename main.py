import streamlit as st
import cv2
import joblib
import config
from pose_detector import PoseDetector
from feature_extractor import FeatureExtractor
from alert_system import AlertSystem
import os
import threading
import subprocess
import socket

# Page Config
st.set_page_config(page_title="Fall Detection System", page_icon="⚠️", layout="wide")

# Helper functions
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_n8n_background():
    if getattr(config, 'USE_N8N', False):
        if is_port_in_use(5678):
            st.sidebar.success("n8n Automation Engine is running (Port 5678).")
        else:
            st.sidebar.info("Starting n8n Automation Engine in the background...")
            try:
                subprocess.Popen("npx n8n start", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                st.sidebar.warning(f"Failed to auto-start n8n: {e}")

class IPVideoStream:
    """Background threaded IP camera stream to prevent buffer lag"""
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stopped = True

    def read(self):
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()

@st.cache_resource
def load_model_cached():
    if not os.path.exists(config.MODEL_PATH):
        return None
    return joblib.load(config.MODEL_PATH)

# Inject custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Automatic Fall Detection & SOS Alert System")

# Sidebar settings
st.sidebar.header("Configuration")
run_app = st.sidebar.checkbox("Start Detection", value=False)
camera_url = st.sidebar.text_input("Camera Index / IP URL", value=str(config.CAMERA_INDEX))
use_n8n = st.sidebar.checkbox("Enable N8N Automation", value=config.USE_N8N)
use_ai = st.sidebar.checkbox("Enable AI Agent Alerts", value=getattr(config, 'USE_AI_AGENT', False))

# Apply config overrides
config.USE_N8N = use_n8n
config.USE_AI_AGENT = use_ai
# Convert camera_url to int if it's purely a digit (local webcam ID)
try:
    config.CAMERA_INDEX = int(camera_url)
except ValueError:
    config.CAMERA_INDEX = camera_url

# Initialize dynamic placeholders
stframe = st.empty()
status_placeholder = st.empty()


if run_app:
    start_n8n_background()
    model = load_model_cached()

    if model is None:
        st.error(f"Error: Trained model '{config.MODEL_PATH}' not found! Run train_model.py first.")
    else:
        # Initialize objects once
        pose_detector = PoseDetector()
        alert_system = AlertSystem()

        # Connect to camera feed
        cap = IPVideoStream(config.CAMERA_INDEX).start()

        if not cap.isOpened():
            st.error("Critical Error: Unable to access the webcam stream. Please check your Camera Index/URL.")
        else:
            tracked_people = {}

            # While loop to constantly output video
            while run_app:
                success, img = cap.read()
                if not success:
                    st.error("Video frame read failure. Possibly end of stream or disconnected camera.")
                    break

                # Resize image for fast YOLO inference
                img = cv2.resize(img, (640, 480))

                img = pose_detector.find_pose(img, draw=True)
                tracked_persons_data = pose_detector.get_landmarks(img)

                persons_in_frame = len(tracked_persons_data)
                any_fall_detected = False

                for person_data in tracked_persons_data:
                    track_id = person_data['id']
                    landmarks = person_data['landmarks']
                    
                    if track_id not in tracked_people:
                        tracked_people[track_id] = {
                            'extractor': FeatureExtractor(),
                            'current_state': 'Standing'
                        }
                    
                    person = tracked_people[track_id]

                    if landmarks:
                        avg_conf = sum(lm[3] for lm in landmarks) / len(landmarks)
                        if avg_conf < 0.5:
                            continue

                        features = person['extractor'].extract_features(landmarks)

                        if features is not None:
                            tilt_angle, hw_ratio, vertical_hip_pos, hip_speed, total_movement = features
                            previous_state = person['current_state']

                            prediction = model.predict([features])[0]
                            
                            # Overrides based on config values
                            if abs(tilt_angle) < config.UPRIGHT_TILT_DEG:
                                if vertical_hip_pos >= config.UPRIGHT_HIP_Y:
                                    prediction = 1
                                else:
                                    prediction = 0
                            
                            head_speed = getattr(person['extractor'], 'head_speed', 0.0)

                            if prediction == 0:
                                person['current_state'] = "Standing"
                            elif prediction == 1:
                                person['current_state'] = "Seated"
                            elif prediction == 2:
                                if previous_state == "Seated":
                                    if head_speed > config.FALL_HEAD_SPEED_THRESHOLD:
                                        person['current_state'] = "Fall"
                                    else:
                                        person['current_state'] = "Sleeping"
                                elif previous_state == "Sleeping":
                                    if total_movement > config.SLEEP_MOVEMENT_THRESHOLD:
                                        person['current_state'] = "Sleeping"
                                    else:
                                        person['current_state'] = "Fall"
                                else:
                                    person['current_state'] = "Fall"
                            elif prediction == 3:
                                if total_movement > config.SLEEP_MOVEMENT_THRESHOLD:
                                    person['current_state'] = "Sleeping"
                                else:
                                    person['current_state'] = "Fall"

                            # Determine trigger
                            if person['current_state'] == "Fall" and previous_state != "Fall":
                                is_real_drop = (head_speed > config.FALL_HEAD_SPEED_THRESHOLD) or (hip_speed > config.FALL_HIP_SPEED_THRESHOLD)
                                if is_real_drop:
                                    is_fast_fall = head_speed > (config.FALL_HEAD_SPEED_THRESHOLD * 1.5)
                                    alert_system.send_alert(is_fast_fall=is_fast_fall)

                            if person['current_state'] == "Fall":
                                any_fall_detected = True

                # Dashboard statuses
                if any_fall_detected:
                    status_placeholder.error(f"⚠️ ALARM: FALL DETECTED | Persons: {persons_in_frame}")
                    cv2.rectangle(img, (20, 20), (550, 80), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, f"ALARM: FALL DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                elif persons_in_frame > 0:
                    status_placeholder.success(f"System Normal. Active Persons: {persons_in_frame}")
                    cv2.rectangle(img, (20, 20), (550, 80), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, f"System: Normal", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                else:
                    status_placeholder.info("System Normal. Awaiting Persons...")

                # Update frame in Streamlit (convert BGR to RGB for Streamlit rendering properly)
                stframe.image(img, channels="BGR", use_column_width=True)

            # Clean-up when stopped
            cap.release()
            st.success("Camera stream successfully closed.")
else:
    st.info("👈 Check 'Start Detection' in the sidebar to begin.")
