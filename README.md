# Vangaurd-Vision -In progress
https://vangaurd-vision.streamlit.app/

## Vanguard Vision: Edge-AI Safety Auditor
Real-Time Proximity Modeling & LLM-Augmented Compliance

# Project Overview
Vanguard Vision is an industrial safety solution designed to mitigate workplace accidents through real-time visual auditing. While its predecessor, Sentinel Dispatch, focused on reactive emergency logistics, Vanguard Vision operates at the "Edge" to proactively prevent collisions between personnel and heavy machinery.

By combining Computer Vision (YOLOv11), Spatial Heuristics, and Generative AI (Groq), the system identifies PPE violations and proximity breaches, automatically generating natural language incident reports for site supervisors.

# System Architecture
The project is built on a three-tier "Perception-to-Reasoning" pipeline:

Perception Layer: A fine-tuned YOLOv11-Nano model trained over 25 epochs on construction-site datasets to identify workers, machinery, and safety gear.

Analytics Layer (Spatial Heuristics): A custom Python engine that calculates Euclidean Distance between detected object centroids to identify high-risk "Zone Breaches."

Reasoning Layer: An integration with the Groq API (Llama 3.3) that transforms raw detection telemetry into structured, human-readable safety audits.

# Tech Stack
Core AI: YOLOv11 (Ultralytics)

LLM Engine: Llama 3.3-70B via Groq Cloud (Low-latency Inference)

Frontend: Streamlit

Data Science: OpenCV, NumPy, PyTorch

Environment: Trained on Google Colab (T4 GPU), Optimized for Edge Deployment.

# Key Technical Highlights
Edge Optimization: Utilized a "Nano" model architecture to achieve sub-50ms inference speeds, making the system viable for on-site hardware like NVIDIA Jetson.

Automated Auditing: Eliminated manual log entries by using LLMs to interpret visual scenes and provide "Corrective Action" recommendations.

Dynamic Risk Thresholds: Implemented a user-configurable "Safety Buffer" via a Streamlit sidebar to allow for variable site conditions.

# Repository Structure

├── app.py                # Main Streamlit Dashboard
├── best.pt               # Trained YOLOv11 Weights (The "Brain")
├── requirements.txt      # Dependencies for Deployment
├── README.md             # Project Documentation
└── sample_data/          # Test images for Proximity Breaches
--
# How to Run
Clone the repo: git clone https://github.com/your-username/vanguard-vision.git

Install requirements: pip install -r requirements.txt

Add your Groq Key: Set your GROQ_API_KEY in your environment secrets.

Launch: streamlit run app.py

# Model Training Performance
The model was trained on a construction-safety dataset with the following final metrics:

Epochs: 25


Primary Classes: Person, Machinery, Hard-hat, Safety Vest.

💡 Portfolio Context
This project demonstrates my ability as a Data Analytics Engineer to move beyond static data and architect end-to-end AI systems that solve high-stakes physical safety problems.
