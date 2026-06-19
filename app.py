import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import tempfile
import json
from pathlib import Path
from PIL import Image

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
WINDOW_SIZE = 30
WINDOW_STEP = 5
DOWNSAMPLE_STEP = 3
SEQ_LEN = 10
ROI_SIZE = 224
N_LANDMARKS = 21
N_COORDS = 3
N_FEATURES = N_LANDMARKS * N_COORDS  # 63
MARGIN = 0.10

# Hardcoded label mapping from label_map.json (fallback)
IDX_TO_LABEL = {
    0: "label15", 1: "label16", 2: "label17", 3: "label18", 4: "label19",
    5: "label1", 6: "label20", 7: "label21", 8: "label22", 9: "label23",
    10: "label24", 11: "label25", 12: "label26", 13: "label27", 14: "label28",
    15: "label29", 16: "label2", 17: "label30", 18: "label31", 19: "label3",
    20: "label4", 21: "label5", 22: "label6", 23: "label7", 24: "label8",
    25: "label9", 26: "label0", 27: "label10", 28: "label11", 29: "label12",
    30: "label13", 31: "label14"
}

# Streamlit Page configuration
st.set_page_config(
    page_title="BISINDO Sign Language Detection",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium styling (dark mode, glassmorphism, Outfit font)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Dark theme override */
        .main {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        
        /* Custom Header Styling */
        .title-gradient {
            background: linear-gradient(135deg, #38BDF8, #818CF8, #C084FC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        
        .subtitle-text {
            color: #94A3B8;
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Glassmorphic card styling */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            margin-bottom: 20px;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0B0F19;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #0284C7, #4F46E5);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
            background: linear-gradient(135deg, #0369A1, #4338CA);
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 800;
            color: #38BDF8;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_yolo_model(model_name):
    """Load the YOLOv8 model and cache it."""
    if not os.path.exists(model_name):
        return None
    return YOLO(model_name)

@st.cache_resource
def load_lstm_model(model_path):
    """Load the Stacked LSTM model and cache it."""
    if not os.path.exists(model_path):
        return None
    try:
        # Load the Keras H5 model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading LSTM model: {str(e)}")
        return None

@st.cache_resource
def get_label_mapping():
    """Load the label map from JSON if exists, otherwise fallback to hardcoded."""
    if os.path.exists("label_map.json"):
        try:
            with open("label_map.json", "r") as f:
                data = json.load(f)
            # Parse idx2label keys as ints
            idx2label = {int(k): v for k, v in data["idx2label"].items()}
            return idx2label
        except Exception:
            return IDX_TO_LABEL
    return IDX_TO_LABEL

# ==========================================
# PIPELINE HELPER FUNCTIONS
# ==========================================
def crop_roi(img_bgr, bbox):
    """Crop the hand ROI from bounding box and resize to 224x224."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    roi = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (ROI_SIZE, ROI_SIZE))

def extract_landmarks(roi_bgr, detector):
    """Extract 21 hand landmarks (63 coordinates) from the ROI."""
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if not res.multi_hand_landmarks:
        return np.zeros(N_FEATURES, dtype=np.float32)
    lm = res.multi_hand_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()

def build_sliding_windows(lm_array, window_size=30, step=5):
    """Construct sliding windows from the timeline of landmark vectors."""
    T = len(lm_array)
    windows = []
    if T < window_size:
        pad = np.zeros((window_size - T, N_FEATURES), dtype=np.float32)
        windows.append(np.vstack([lm_array, pad]))
    else:
        for start in range(0, T - window_size + 1, step):
            windows.append(lm_array[start : start + window_size])
    return windows

def downsample_30_to_10(window_30x63):
    """Downsample a 30-frame window to 10 frames by selecting every 3rd frame."""
    return window_30x63[::DOWNSAMPLE_STEP]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ==========================================
# WEB UI LAYOUT
# ==========================================
st.markdown("<div class='title-gradient'>Deteksi Bahasa Isyarat BISINDO</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Pipeline Pengenalan Bahasa Isyarat Hibrida YOLOv8s + MediaPipe + Stacked LSTM</div>", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.markdown("### ⚙️ Konfigurasi Model")

# Choose YOLO weight
yolo_weight_options = ["best.pt", "last.pt"]
selected_yolo_weight = st.sidebar.selectbox(
    "Bobot YOLO (Deteksi Tangan)",
    yolo_weight_options,
    help="Pilih file bobot YOLOv8s hasil training untuk deteksi tangan."
)

# Threshold sliders
conf_thresh = st.sidebar.slider(
    "YOLO Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Ambang batas keyakinan minimum untuk bounding box tangan YOLO."
)

iou_thresh = st.sidebar.slider(
    "YOLO IoU Threshold (NMS)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="IoU Threshold untuk Non-Maximum Suppression."
)

# Load label mapping
label_map = get_label_mapping()

# Load models
yolo_model = load_yolo_model(selected_yolo_weight)
lstm_model = load_lstm_model("lstm_best.h5")

# Sidebar status cards
st.sidebar.markdown("### 📊 Status Sistem")
if yolo_model:
    st.sidebar.success(f"YOLOv8s ({selected_yolo_weight}) Loaded")
else:
    st.sidebar.error(f"YOLOv8s ({selected_yolo_weight}) Not Found in directory")

if lstm_model:
    st.sidebar.success("Stacked LSTM (lstm_best.h5) Loaded")
else:
    st.sidebar.error("Stacked LSTM (lstm_best.h5) Not Found in directory")

st.sidebar.markdown(
    """
    <div class='glass-card' style='padding: 15px; margin-top: 15px;'>
        <h4 style='margin-top:0; color:#38BDF8; font-size:1rem;'>Alur Pipeline:</h4>
        <ol style='margin-bottom:0; font-size:0.85rem; padding-left:15px; color:#94A3B8;'>
            <li>Upload File Video</li>
            <li>Downsample frame rate ke 10 FPS</li>
            <li>Deteksi Tangan via YOLOv8s</li>
            <li>Crop ROI & Resize (224x224)</li>
            <li>Ekstraksi 21 Landmark MediaPipe</li>
            <li>Sliding Window 30 → 10 (Downsample [::3])</li>
            <li>Klasifikasi 32 Kelas via Stacked LSTM</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

# Main container
if not yolo_model or not lstm_model:
    st.warning("Silakan pastikan file model `best.pt`/`last.pt` dan `lstm_best.h5` berada dalam folder yang sama dengan skrip ini.")
else:
    # Video upload card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Video Bahasa Isyarat")
    uploaded_file = st.file_uploader(
        "Pilih file video (Format: MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Unggah video gerakan bahasa isyarat BISINDO satu kata."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Save uploaded file to temp file to read via OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        # Open video and read properties
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Close capture for now
        cap.release()

        # Display video details
        col_prop1, col_prop2, col_prop3, col_prop4 = st.columns(4)
        with col_prop1:
            st.metric("Resolusi Video", f"{width}x{height}")
        with col_prop2:
            st.metric("FPS Asli", f"{fps:.2f} fps")
        with col_prop3:
            st.metric("Durasi Video", f"{total_frames / max(fps, 1):.2f} detik")
        with col_prop4:
            st.metric("Total Frame", f"{total_frames} frame")

        st.markdown("<br>", unsafe_allow_html=True)

        # Trigger processing button
        if st.button("Mulai Proses Deteksi"):
            st.markdown("### 🖐️ Pemrosesan Frame Live & Deteksi")
            
            # Setup columns for live display
            col_live1, col_live2 = st.columns([2, 1])
            with col_live1:
                st.markdown("##### Frame Utama + Overlay Deteksi")
                main_frame_placeholder = st.empty()
            with col_live2:
                st.markdown("##### Cropped ROI & Landmark")
                roi_frame_placeholder = st.empty()

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize MediaPipe Hands detector locally for thread safety
            hand_detector = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )

            # Reopen video
            cap = cv2.VideoCapture(tfile.name)
            
            # Downsampling frame step
            target_fps = 10
            step = max(int(round(fps / target_fps)), 1)

            frame_idx = 0
            processed_frames_count = 0
            all_landmarks = []
            
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract only at 10 FPS
                if frame_idx % step == 0:
                    processed_frames_count += 1
                    status_text.text(f"Memproses frame ke-{processed_frames_count} (Frame Asli: {frame_idx + 1}/{total_frames})...")
                    
                    # 1. Run YOLOv8s hand detection
                    results = yolo_model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
                    
                    roi = None
                    bbox = None
                    
                    # If hands detected, crop ROI
                    if len(results.boxes) > 0:
                        # Take the first hand with highest confidence
                        bbox = results.boxes.xyxy.cpu().numpy()[0]
                        roi = crop_roi(frame, bbox)
                    
                    # 2. Extract MediaPipe landmarks
                    if roi is None:
                        # Fallback to zero vector if YOLO doesn't detect
                        vec = np.zeros(N_FEATURES, dtype=np.float32)
                        # Display raw frame on main placeholder
                        annotated_frame = frame.copy()
                        main_frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Display gray image for ROI placeholder
                        black_roi = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
                        cv2.putText(black_roi, "Tangan Tidak Terdeteksi", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        roi_frame_placeholder.image(black_roi, use_container_width=True)
                    else:
                        # Extract landmarks
                        vec = extract_landmarks(roi, hand_detector)
                        
                        # Render visual overlay on frame for user review
                        annotated_frame = frame.copy()
                        # Draw YOLO Bounding Box
                        bx1, by1, bx2, by2 = bbox.astype(int)
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                        cv2.putText(annotated_frame, "Tangan", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Draw MediaPipe keypoints on cropped ROI
                        roi_annotated = roi.copy()
                        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        res_roi = hand_detector.process(rgb_roi)
                        if res_roi.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                roi_annotated,
                                res_roi.multi_hand_landmarks[0],
                                mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style()
                            )
                        
                        # Display visual outputs
                        main_frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        roi_frame_placeholder.image(cv2.cvtColor(roi_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                    all_landmarks.append(vec)

                    # Update progress bar
                    prog = min(int((frame_idx / total_frames) * 100), 100)
                    progress_bar.progress(prog)

                frame_idx += 1

            cap.release()
            hand_detector.close()
            
            processing_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.text(f"Selesai memproses {processed_frames_count} frame dalam {processing_time:.2f} detik!")

            # 3. Sliding Window Inference
            if len(all_landmarks) > 0:
                lm_array = np.array(all_landmarks, dtype=np.float32)
                windows30 = build_sliding_windows(lm_array, WINDOW_SIZE, WINDOW_STEP)
                
                # Downsample 30 to 10
                windows10 = [downsample_30_to_10(w) for w in windows30]
                X_input = np.array(windows10, dtype=np.float32)  # shape: (num_windows, 10, 63)
                
                # 4. Predict via Stacked LSTM
                predictions = lstm_model.predict(X_input)  # shape: (num_windows, 32)
                
                # Aggregate predictions
                # Get the mean probability across all sliding windows
                mean_predictions = np.mean(predictions, axis=0)
                best_class_idx = np.argmax(mean_predictions)
                best_class_name = label_map.get(best_class_idx, f"label{best_class_idx}")
                confidence = mean_predictions[best_class_idx]
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("### 🏆 Hasil Prediksi Bahasa Isyarat")
                
                # Display final results
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Kata Isyarat Terdeteksi", best_class_name.upper())
                with col_res2:
                    st.metric("Tingkat Keyakinan (Confidence)", f"{confidence * 100:.2f}%")
                with col_res3:
                    st.metric("Total Window Dihitung", f"{len(windows30)} window")
                
                # Detailed Dashboard
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("#### 📊 Distribusi Probabilitas Top-5 Kelas Teratas")
                
                # Plot top-5 predictions
                top5_indices = np.argsort(mean_predictions)[::-1][:5]
                top5_labels = [label_map.get(idx, f"label{idx}").upper() for idx in top5_indices]
                top5_probs = mean_predictions[top5_indices] * 100
                
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#1E293B')
                ax.set_facecolor('#1E293B')
                
                bars = ax.barh(top5_labels[::-1], top5_probs[::-1], color='#38BDF8', edgecolor='white', height=0.5)
                ax.set_xlabel('Probability (%)', color='white', fontsize=10)
                ax.set_title('Top-5 Predicted BISINDO Signs', color='white', fontsize=12, fontweight='bold')
                ax.tick_params(colors='white', labelsize=9)
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add text labels on the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                            va='center', ha='left', color='white', fontsize=9, fontweight='bold')
                            
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

                # Predictions timeline card
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("#### ⏳ Kronologi Deteksi Window-by-Window")
                
                # Prepare DataFrame for timeline
                timeline_data = []
                for idx, pred in enumerate(predictions):
                    pred_class_idx = np.argmax(pred)
                    pred_class_name = label_map.get(pred_class_idx, f"label{pred_class_idx}").upper()
                    pred_conf = pred[pred_class_idx]
                    
                    # Convert window index to approximate timestamp
                    # Each window starts at `start = idx * WINDOW_STEP` frames at 10 FPS
                    timestamp = (idx * WINDOW_STEP) / target_fps
                    
                    timeline_data.append({
                        "Window Ke": idx + 1,
                        "Estimasi Detik": f"{timestamp:.2f} s",
                        "Prediksi Kata": pred_class_name,
                        "Confidence Score": f"{pred_conf * 100:.2f}%"
                    })
                
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(df_timeline, use_container_width=True)
                
                # Export option
                csv = df_timeline.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (.csv)",
                    data=csv,
                    file_name="hasil_prediksi_bisindo.csv",
                    mime="text/csv"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.error("Gagal melakukan ekstraksi sekuens landmark. Cek apakah tangan terdeteksi di video.")

        # Clean up temp file
        try:
            os.unlink(tfile.name)
        except OSError:
            pass
