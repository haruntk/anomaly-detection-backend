import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os

# Görüntü dönüştürme işlemleri için transform tanımı
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def threshold_otsu(img_gray):
    """Otsu eşikleme yöntemi ile görüntüyü ikili formata dönüştürür."""
    _, thresh_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh_img

def threshold_adaptive(img_gray):
    """Adaptif eşikleme yöntemi ile görüntüyü ikili formata dönüştürür."""
    thresh_img = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Ters çevrilmiş ikili eşikleme
        11, 
        2    
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh_img

def load_image(file_storage):
    """Dosya depolama nesnesinden görüntüyü yükler ve gri tonlamalı formata dönüştürür."""
    image_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Doğrudan gri tonlamaya dönüştür

def preprocess_image(image, threshold_func=threshold_otsu, remove_background=True, target_size=256):
    """Görüntüyü ön işleme adımlarından geçirir."""
    if isinstance(image, str):
        # Eğer görüntü bir dosya yolu ise, görüntüyü yükle
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Doğrudan gri tonlamalı oku
        if img is None:
            raise ValueError(f"Hata: '{image}' okunamadı.")
    else:
        # Eğer görüntü zaten bir numpy dizisi ise
        if len(image.shape) == 3:  # Renkli görüntü ise
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()

    # Görüntüyü yeniden boyutlandır
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Eşikleme işlemi
    if remove_background:
        thresholded = threshold_func(img)
        
        # Nesne konturu kontrolü - eğer çok az piksel kaldıysa arka planı kaldırma
        white_pixel_count = cv2.countNonZero(thresholded)
        if white_pixel_count < 100:  # Çok az nesne pikselimiz var
            # Orijinal görüntüyü kullan
            print("Uyarı: Arka plan kaldırıldıktan sonra çok az nesne kaldı, orijinal görüntü kullanılıyor.")
            img_processed = img
        else:
            img_processed = thresholded
    else:
        img_processed = img
    
    # Normalize et
    normalized_img = img_processed.astype(np.float32) / 255.0
    
    # 3 kanallı görüntüye dönüştür (gri tonlamalıyı 3 kez tekrarla)
    normalized_img = np.stack([normalized_img] * 3, axis=-1)
    
    return normalized_img

def transform_image(image_np):
    """Görüntüyü PyTorch tensor formatına dönüştürür."""
    return transform(image_np)

def extract_features(extractor, tensor):
    """Görüntüden özellik çıkarır."""
    with torch.no_grad():
        features = extractor(tensor)
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1).cpu().numpy()
    return features

def compute_anomaly_score(features, mean, cov_inv):
    """Anomali skorunu hesaplar."""
    scores = []
    for i in range(features.shape[1]):
        delta = features[:, i, :] - mean[i]
        score = np.einsum('ij,jk,ik->i', delta, cov_inv[i], delta)
        scores.append(score)
    return np.max(np.array(scores), axis=0)

def process_dataset(root_dir, output_root_dir, threshold_func=threshold_otsu, remove_background=True):
    """Veri setindeki tüm görüntüleri işler ve kaydeder."""
    sub_dirs = ['train', 'test', 'ground_truth']

    for sub in sub_dirs:
        current_dir = os.path.join(root_dir, sub)
        if not os.path.exists(current_dir):
            continue  

        for sub_sub in os.listdir(current_dir):
            sub_sub_path = os.path.join(current_dir, sub_sub)
            if not os.path.isdir(sub_sub_path):
                continue

            output_dir = os.path.join(output_root_dir, sub, sub_sub)
            os.makedirs(output_dir, exist_ok=True)

            for img_name in os.listdir(sub_sub_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_img_path = os.path.join(sub_sub_path, img_name)
                    output_img_path = os.path.join(output_dir, img_name)

                    # Görüntüyü işle
                    processed_img = preprocess_image(input_img_path, threshold_func, remove_background)
                    
                    # İşlenmiş görüntüyü kaydet
                    output_img = (processed_img * 255).astype(np.uint8)
                    cv2.imwrite(output_img_path, output_img)