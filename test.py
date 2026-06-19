import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("Numpy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("MediaPipe version:", mp.__version__)
print("Pandas version:", pd.__version__)

try:
    print("Testing mp.solutions.hands...")
    mp_hands = mp.solutions.hands
    print("mp.solutions.hands exists!", mp_hands)
except Exception as e:
    print("mp.solutions error:", e)

try:
    print("Loading YOLO model...")
    yolo = YOLO("best.pt")
    print("YOLO loaded successfully!")
except Exception as e:
    print("YOLO load error:", e)

try:
    print("Loading LSTM model...")
    lstm = tf.keras.models.load_model("lstm_best.h5")
    print("LSTM loaded successfully!")
except Exception as e:
    print("LSTM load error:", e)
