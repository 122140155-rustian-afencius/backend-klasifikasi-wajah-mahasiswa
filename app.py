import os
import json
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from insightface.app import FaceAnalysis
from skimage import transform as trans
import gc

app = Flask(__name__)

DEVICE = torch.device('cpu')
MODEL_PATH = "best_model.pth"
CLASS_PATH = "class_names.json"

with open(CLASS_PATH, 'r') as f:
    class_names = json.load(f)
    
num_classes = len(class_names)

face_app = FaceAnalysis(name='buffalo_l', root='/root/.insightface', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
gc.collect()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def custom_norm_crop(img, landmark, image_size=112, crop_scale=1.0):
    src = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041] ], dtype=np.float32)
    
    if crop_scale != 1.0:
        src = (src - 56) * crop_scale + 56
    if image_size != 112:
        src = src * (image_size / 112.0)
        
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

print("Loading model...")
model = InceptionResnetV1(
    classify=True, 
    num_classes=num_classes, 
    dropout_prob=0.5,
    pretrained=None
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.to(DEVICE)
model.eval()
gc.collect()
print("Model loaded!")


@app.route('/', methods=['GET'])
def index():
    return "Face Recognition API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        faces = face_app.get(img_bgr)
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        target_face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
        
        face_crop = custom_norm_crop(img_bgr, target_face.kps, image_size=224, crop_scale=0.8)
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        input_tensor = val_tf(face_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            
            top5_prob, top5_idx = torch.topk(probs, 5)
            
        results = []
        top5_prob = top5_prob.cpu().numpy()[0]
        top5_idx = top5_idx.cpu().numpy()[0]
        
        for i in range(5):
            results.append({
                "rank": i + 1,
                "class": class_names[top5_idx[i]],
                "confidence": float(f"{top5_prob[i]:.4f}"),
                "confidence_percent": f"{top5_prob[i]*100:.2f}%"
            })

        return jsonify({
            "message": "Success",
            "predictions": results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)