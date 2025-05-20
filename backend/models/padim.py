import torch
import numpy as np
from torchvision import models
from utils.preprocess import transform_image, extract_features, compute_anomaly_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load weights
mean = np.load("weights/padim/padim_mean.npy")
cov_inv = np.load("weights/padim/padim_cov_inv.npy")

# feature extractor
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-2])
feature_extractor.to(device).eval()

def predict(processed_image):
    """
    Ön işlenmiş görüntüyü kullanarak anomali tespiti yapar.
    
    Args:
        processed_image: Ön işlenmiş (normalize edilmiş, arka planı kaldırılmış) görüntü
        
    Returns:
        dict: Tespit sonucu ve anomali skoru
    """
    # Görüntüyü tensor formatına dönüştür
    image_tensor = transform_image(processed_image).unsqueeze(0).to(device)
    
    # Özellik çıkar
    features = extract_features(feature_extractor, image_tensor)
    
    # Anomali skorunu hesapla
    score = compute_anomaly_score(features, mean, cov_inv)
    
    # Sonucu belirle
    result = "defect" if score > 162966574.90122232 else "good"
    
    return {
        "result": result,
        "score": float(score),
        "message": "Kusurlu" if result == "defect" else "Sağlam"
    }