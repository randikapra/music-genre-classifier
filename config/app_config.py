"""
Configuration settings for the music genre classifier application.
"""

# Spark configuration
SPARK_CONFIG = {
    "app_name": "Music Genre Classification",
    "driver_memory": "6g",
    "executor_memory": "6g"
}

import os
import subprocess
from pathlib import Path
# Get the base directory (to make paths relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration settings with relative paths
DATASET_PATH = os.path.join(BASE_DIR, "data/Merged_dataset.csv")
MODEL_BASE_PATH = os.path.join(BASE_DIR, "models/music_genre_classifier")
# Data paths
# DATASET_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/merged_music_dataset.csv"
# MODEL_BASE_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier"

# Model file paths
RF_MODEL_PATH = f"{MODEL_BASE_PATH}_rf"
GBT_MODEL_PATH = f"{MODEL_BASE_PATH}_gbt"
LR_MODEL_PATH = f"{MODEL_BASE_PATH}_lr"
LABEL_MAPPING_PATH = f"{MODEL_BASE_PATH}_label_mapping"

# Training parameters
TRAIN_TEST_SPLIT = [0.8, 0.2]
RANDOM_SEED = 42

# Class balancing parameters
MIN_SAMPLES = 1000
MAX_SAMPLES = 2000

# Custom stopwords for music domain
MUSIC_STOP_WORDS = [
    "oh", "yeah", "hey", "la", "na", "mmm", "ooh", "ah", "baby", "chorus", "verse", 
    "repeat", "instrumental", "guitar", "solo", "intro", "outro", "uh", "whoa"
]

# Web application settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
DEBUG_MODE = True

# Chart colors for visualization
CHART_COLORS = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", 
    "#9966FF", "#FF9F40", "#8C9EFF"
]