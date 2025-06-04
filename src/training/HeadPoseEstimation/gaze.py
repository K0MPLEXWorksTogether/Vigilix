import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load your custom YOLOv11 face detection model
face_model = YOLO("path/to/your/yolov11-face.pt")

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Indices for iris and eye landmarks in MediaPipe
LEFT_EYE_LANDMARKS = [33, 133]      # Approx corners of left eye
RIGHT_EYE_LANDMARKS = [362, 263]    # Approx corners of right eye
LEFT_IRIS = [468]
RIGHT_IRIS = [473]

def get_gaze_direction(eye_landmarks, iris_landmark):
    """Estimate if gaze is left, right, or center based on iris position"""
    x1, x2 = eye_landmarks[0][0], eye_landmarks[1][0]
    iris_x = iris_landmark[0]
    midpoint = (x1 + x2) / 2
    eye_width = abs(x2 - x1)

    if iris_x < x1 + 0.35 * eye_width:
        return "Right"
    elif iris_x > x1 + 0.65 * eye_width:
        return "Left"
    else:
        return "Center"

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face with YOLOv11
    results = face_model(frame)
    detections = results[0].boxes

    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        face_roi = frame[y1:y2, x1:x2]
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Run MediaPipe on the cropped face
        result = face_mesh.process(rgb_face)
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            h, w, _ = face_roi.shape

            # Get eye and iris positions
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_LANDMARKS]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_LANDMARKS]
            left_iris = (int(face_landmarks.landmark[LEFT_IRIS[0]].x * w), int(face_landmarks.landmark[LEFT_IRIS[0]].y * h))
            right_iris = (int(face_landmarks.landmark[RIGHT_IRIS[0]].x * w), int(face_landmarks.landmark[RIGHT_IRIS[0]].y * h))

            # Estimate gaze
            left_gaze = get_gaze_direction(left_eye, left_iris)
            right_gaze = get_gaze_direction(right_eye, right_iris)

            # Majority vote or just show one eye
            gaze = left_gaze if left_gaze == right_gaze else "Center"

            # Draw gaze direction
            cv2.putText(frame, f"Gaze: {gaze}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Gaze Direction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
