import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os

# Define transform for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def threshold_otsu(img_gray):
    """Convert image to binary format using Otsu thresholding."""
    _, thresh_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh_img

def threshold_adaptive(img_gray):
    """Convert image to binary format using adaptive thresholding."""
    thresh_img = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Inverted binary thresholding
        11, 
        2    
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh_img

def load_image(file_storage):
    """Load image from file storage and convert to grayscale."""
    image_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Directly convert to grayscale

def preprocess_image(image, threshold_func=threshold_otsu, remove_background=True, target_size=256):
    """Process image through preprocessing steps."""
    if isinstance(image, str):
        # If image is a file path, load the image
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Directly read grayscale
        if img is None:
            raise ValueError(f"Hata: '{image}' okunamadı.")
    else:
        # If image is already a numpy array
        if len(image.shape) == 3:  # If image is color
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()

    # Resize image
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Thresholding
    if remove_background:
        thresholded = threshold_func(img)
        
        # Object contour check - if too few pixels remain, remove background
        white_pixel_count = cv2.countNonZero(thresholded)
        if white_pixel_count < 100:  # Too few object pixels remain
            # Use original image
            print("Warning: Too few object pixels remain after background removal, using original image.")
            img_processed = img
        else:
            img_processed = thresholded
    else:
        img_processed = img
    
    # Normalize
    normalized_img = img_processed.astype(np.float32) / 255.0
    
    # Convert to 3 channel image (repeat grayscale 3 times)
    normalized_img = np.stack([normalized_img] * 3, axis=-1)
    
    return normalized_img

def transform_image(image_np):
    """Convert image to PyTorch tensor format."""
    return transform(image_np)

def extract_features(extractor, tensor):
    """Extract features from image."""
    with torch.no_grad():
        features = extractor(tensor)
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1).cpu().numpy()
    return features

def process_dataset(root_dir, output_root_dir, threshold_func=threshold_otsu, remove_background=True):
    """Process all images in the dataset and save them."""
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

                    # Process image
                    processed_img = preprocess_image(input_img_path, threshold_func, remove_background)
                    
                    # Save processed image
                    output_img = (processed_img * 255).astype(np.uint8)
                    cv2.imwrite(output_img_path, output_img)