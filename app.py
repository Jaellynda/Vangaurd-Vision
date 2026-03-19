import streamlit as st
import cv2
import math
import os
import numpy as np
from ultralytics import YOLO
from groq import Groq
from PIL import Image

# --- 1. SETTINGS & THEME ---
st.set_page_config(page_title="Vanguard Vision AI", page_icon="🛡️", layout="wide")

# Custom CSS for a professional "Edge AI" look
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .stStatusWidget { visibility: hidden; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 2. CONFIGURATION ---
# This matches your specific Colab training run path
MODEL_PATH = "/content/runs/detect/Vanguard_Vision_v12/weights/best.pt"
# If running locally, change this to "best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "best.pt" 

# Replace with your actual Groq API Key
GROQ_API_KEY = "your_groq_api_key_here" 

# --- 3. INITIALIZE MODELS ---
@st.cache_resource
def load_vanguard_brain():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

vanguard_model = load_vanguard_brain()
groq_client = Groq(api_key=GROQ_API_KEY)

# --- 4. ANALYTICS ENGINE (Proximity Logic) ---
def get_distance(box1, box2):
    """Calculates Euclidean distance between centers of two bounding boxes."""
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# --- 5. DASHBOARD UI ---
st.title("🛡️ Vanguard Vision: Edge-AI Safety Auditor")
st.subheader("Real-Time Proximity Modeling & PPE Compliance")

# Sidebar for Professional Interaction
st.sidebar.header("🕹️ Control Panel")
dist_threshold = st.sidebar.slider("Proximity Alert Threshold (Pixels)", 50, 500, 200)
conf_level = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.info("Vanguard Vision complements **Sentinel Dispatch** by providing real-time on-site monitoring.")

# Main Upload Component
uploaded_file = st.file_uploader("Upload Construction Site Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    if vanguard_model is None:
        st.error(f"Brain file not found at {MODEL_PATH}. Please upload 'best.pt' to the directory.")
    else:
        # Convert file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        
        # Phase 1: Computer Vision Inference
        results = vanguard_model.predict(opencv_img, conf=conf_level)
        annotated_frame = results[0].plot()
        
        # Phase 2: Spatial Analytics (Proximity Check)
        detections = results[0].boxes
        people = [b.xyxy[0].tolist() for b in detections if vanguard_model.names[int(b.cls)] == 'person']
        machinery = [b.xyxy[0].tolist() for b in detections if vanguard_model.names[int(b.cls)] in ['machinery', 'vehicle', 'excavator']]
        
        breach_detected = False
        for p in people:
            for m in machinery:
                if get_distance(p, m) < dist_threshold:
                    breach_detected = True
                    break

        # Layout for Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(annotated_frame, channels="BGR", caption="Inference Stream: Vanguard Vision v1.2")
            
        with col2:
            st.write("### 🚨 Safety Audit Status")
            if breach_detected:
                st.error("CRITICAL BREACH: Proximity Violation!")
            else:
                st.success("SITE SECURE: No Proximity Violations")
            
            # Show specific counts
            classes_found = [vanguard_model.names[int(b.cls)] for b in detections]
            st.write(f"**Objects Detected:** {len(classes_found)}")
            for cls in set(classes_found):
                st.write(f"- {cls}: {classes_found.count(cls)}")

            # Phase 3: Generative AI Reasoning (Groq)
            st.markdown("---")
            if st.button("Generate AI Safety Incident Report"):
                with st.spinner("Llama 3.3 Auditor is analyzing site telemetry..."):
                    prompt = f"""
                    Role: Industrial Safety Auditor AI.
                    Telemetry Data: {classes_found}.
                    Proximity Violation Detected: {'YES' if breach_detected else 'NO'}.
                    Task: Write a concise, 3-sentence incident report for a Data Engineering portfolio. 
                    Focus on risk mitigation and compliance.
                    """
                    try:
                        chat_completion = groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile",
                        )
                        st.info(chat_completion.choices[0].message.content)
                    except Exception as e:
                        st.warning("Connect Groq API Key to enable AI Reporting.")

st.markdown("---")
st.caption("Data Analytics Engineering Portfolio | Built by [Your Name] | 2026")
