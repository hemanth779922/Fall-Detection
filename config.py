# Configuration settings for the Fall Detection System

import os

# Camera settings
# Replace with your phone's IP Webcam URL (e.g., using "IP Webcam" Android app)
# Make sure to append '/video' at the end of the URL for the video stream.
CAMERA_INDEX = "http://192.168.1.5:8080/video"  # User's provided IP Camera URL

# Alert System Settings
SEND_REAL_SMS = True  # Set to True to use Twilio, False for mock SMS printed to terminal
ALERT_PHONE_NUMBER = "+917799224371"  # Predefined mobile number for SOS

# API Keys and Credentials
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_FROM_NUMBER = ""

# n8n Integration Settings
USE_N8N = False  # Turned off since the webhook is currently returning 404
N8N_WEBHOOK_URL = "https://kvignan77.app.n8n.cloud/webhook-test/51fd89ef-dca5-432f-9df3-5f3865093e70" # Switched back to Test Endpoint

# AI Agent Integration Settings
USE_AI_AGENT = False # Set to True to use LangChain AI Agent instead of n8n
OPENAI_API_KEY = "" 
ANTHROPIC_API_KEY = ""

# Google API key (for Google services if/when integrated)
GOOGLE_API_KEY = ""

# Model settings
MODEL_PATH = "fall_model.pkl"
YOLO_MODEL_PATH = "yolov8n-pose.pt"
DATASET_PATH = "fall_dataset.csv"  # Path to training data

# Thresholds/Parameters
FPS_DELAY = 1  # Delay for cv2.waitKey()

# Posture / fall logic thresholds (used in main.py)
# Frames to keep in short history when checking recent posture
POSTURE_HISTORY_FRAMES = 15

# A frame is considered "upright / standing" if:
UPRIGHT_TILT_DEG = 45.0      # body axis close to vertical
UPRIGHT_HIP_Y = 0.6          # hip not too close to floor (0=top, 1=bottom)

# To treat something as a *real fall* we require:
FALL_HIP_SPEED_THRESHOLD = 0.12  # sudden hip drop between frames
FALL_HEAD_SPEED_THRESHOLD = 0.08 # sudden head drop (useful from sitting/sleeping positions)

# Movement threshold for a sleeping person (if they are tossing/turning)
SLEEP_MOVEMENT_THRESHOLD = 0.05
