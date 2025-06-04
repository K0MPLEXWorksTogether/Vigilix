import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import time
import webrtcvad
import collections
from datetime import datetime

# Setup
os.makedirs("evidence/photo", exist_ok=True)
os.makedirs("evidence/audio", exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

SAMPLE_RATE = 16000  # Required by webrtcvad
DURATION = 1  # For VAD check
EVIDENCE_DURATION = 2  # Longer audio to store if valid
VAD_FRAME_DURATION = 30  # ms

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0–3 (aggressiveness): 3 = most aggressive

def is_mouth_open(landmarks, image_shape):
    h, w, _ = image_shape
    upper = landmarks[UPPER_LIP_IDX]
    lower = landmarks[LOWER_LIP_IDX]
    return abs((lower.y - upper.y) * h) > 10  # Tune this threshold

def record_audio_raw(duration=DURATION):
    return sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')

def vad_check(audio):
    # Convert audio to bytes in 10-30ms frames
    audio_bytes = audio.tobytes()
    frame_length = int(SAMPLE_RATE * VAD_FRAME_DURATION / 1000) * 2  # 16-bit mono = 2 bytes
    speech_frames = 0
    for i in range(0, len(audio_bytes), frame_length):
        frame = audio_bytes[i:i + frame_length]
        if len(frame) == frame_length and vad.is_speech(frame, SAMPLE_RATE):
            speech_frames += 1
    return speech_frames > 0

def record_evidence_audio(path):
    print("Recording evidence audio...")
    audio = sd.rec(int(EVIDENCE_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(path, SAMPLE_RATE, audio)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            if is_mouth_open(landmarks, frame.shape):
                audio_clip = record_audio_raw()
                sd.wait()
                if vad_check(audio_clip):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    photo_path = f"evidence/photo/photo_{timestamp}.jpg"
                    audio_path = f"evidence/audio/audio_{timestamp}.wav"

                    cv2.imwrite(photo_path, frame)
                    record_evidence_audio(audio_path)

                    print(f"[✔] Evidence saved: {photo_path}, {audio_path}")
                else:
                    print("[!] Sound detected, but not human speech.")

    cv2.imshow("Monitoring", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
