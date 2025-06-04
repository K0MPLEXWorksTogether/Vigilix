import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv11 model
yolo_model = YOLO('../../../models/face/face-detection.pt')  # Your trained face detection model

# Load FaceNet model (pretrained on VGGFace2)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_embeddings(image_path):
    # Load image using OpenCV
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces using YOLOv11
    results = yolo_model(img_rgb)

    embeddings = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # x1, y1, x2, y2

        for (x1, y1, x2, y2) in boxes:
            # Crop and preprocess face
            face = img_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face).resize((160, 160))

            # Convert to tensor
            face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
            face_tensor = (face_tensor - 127.5) / 128.0  # Normalize

            # Get embedding
            with torch.no_grad():
                embedding = facenet(face_tensor)
            embeddings.append(embedding.squeeze().numpy())

            # Optionally show the face
            plt.imshow(face_pil)
            plt.title("Detected Face")
            plt.axis("off")
            plt.show()

    return embeddings

# Example usage
if __name__ == '__main__':
    image_path = 'person.jpg'  # Replace with your image
    embeddings = extract_face_embeddings(image_path)

    if embeddings:
        print("Extracted Face Embeddings:")
        for i, emb in enumerate(embeddings):
            print(f"Face {i + 1}:")
            print(emb)
            print(f"Shape: {emb.shape}")
