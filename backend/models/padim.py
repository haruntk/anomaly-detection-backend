import torch
import numpy as np
from torchvision import models
from utils.preprocess import transform_image, extract_features

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
    Predict anomaly detection using preprocessed image.
    
    Args:
        processed_image: Preprocessed (normalized, background removed) image
        
    Returns:
        dict: Prediction result and anomaly score
    """
    # Convert image to tensor format
    image_tensor = transform_image(processed_image).unsqueeze(0).to(device)
    
    # Extract features
    features = extract_features(feature_extractor, image_tensor)
    
    # Compute anomaly score
    score = compute_anomaly_score(features, mean, cov_inv)

    min_score = 15000000
    max_score = 3500000000

    normalized_score = (score - min_score) / (max_score - min_score)
    normalized_threshold = (162966574.90122232 - min_score) / (max_score - min_score)

    # Determine result
    result = "defect" if normalized_score > normalized_threshold else "good"
    
    return {
        "result": result,
        "score": float(normalized_score),
        "message": "Defect" if result == "defect" else "Good"
    }

def compute_anomaly_score(features, mean, cov_inv):
    """Anomali skorunu hesaplar."""
    scores = []
    for i in range(features.shape[1]):
        delta = features[:, i, :] - mean[i]
        score = np.einsum('ij,jk,ik->i', delta, cov_inv[i], delta)
        scores.append(score)
    return np.max(np.array(scores), axis=0)