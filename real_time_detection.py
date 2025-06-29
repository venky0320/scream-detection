import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
import pywhatkit as kit
import geocoder
import tkinter as tk
from threading import Thread

# Load the trained model
model = tf.keras.models.load_model("scream_detector_model.h5")

# Function to extract features from audio data
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Function to get live GPS location and generate Google Maps link
def get_live_location():
    g = geocoder.ip('me')  # Get current location
    if g.latlng:
        latitude, longitude = g.latlng
        maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
        return maps_link
    return "Location unavailable"

# Function to send emergency WhatsApp alert
def send_whatsapp_alert():
    try:
        location_link = get_live_location()
        message = f"🚨 EMERGENCY ALERT: A scream was detected! 📍 Live Location: {location_link}"
        kit.sendwhatmsg_instantly("+9191XXXXXXXX", message, wait_time=10)
        status_label.config(text="WhatsApp Alert Sent ✅", fg="green")
    except Exception as e:
        status_label.config(text=f"WhatsApp Alert Failed ❌: {e}", fg="red")

# Function to process real-time audio
def detect_scream(indata, frames, time, status):
    audio_data = indata[:, 0]  
    features = extract_features(audio_data, 22050)  # Use 22050Hz sample rate

    prediction = model.predict(np.array([features]))[0][0]

    if prediction > 0.5:
        status_label.config(text="🚨 Scream detected! Sending alert...", fg="red")
        send_whatsapp_alert()
    else:
        status_label.config(text="✅ No scream detected", fg="green")

# Function to start real-time detection
def start_detection():
    global stream
    status_label.config(text="🎤 Listening for screams...", fg="blue")
    stream = sd.InputStream(callback=detect_scream, samplerate=22050, channels=1)
    stream.start()
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

# Function to stop detection
def stop_detection():
    global stream
    stream.stop()
    status_label.config(text="Detection Stopped ❌", fg="black")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

# GUI Setup
root = tk.Tk()
root.title("Scream Detection System")
root.geometry("400x300")
root.config(bg="white")

title_label = tk.Label(root, text="🔊 Scream Detection System", font=("Arial", 16, "bold"), fg="black", bg="white")
title_label.pack(pady=10)

status_label = tk.Label(root, text="Click Start to Begin", font=("Arial", 12), fg="gray", bg="white")
status_label.pack(pady=5)

start_button = tk.Button(root, text="Start Detection", font=("Arial", 12), command=lambda: Thread(target=start_detection).start(), bg="green", fg="white", padx=10, pady=5)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", font=("Arial", 12), command=stop_detection, bg="red", fg="white", padx=10, pady=5, state=tk.DISABLED)
stop_button.pack(pady=10)

root.mainloop()
