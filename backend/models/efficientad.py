import torch
import numpy as np
import os
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from torchvision import models, transforms
from utils.preprocess import transform_image, extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model weights paths
AUTOENCODER_PATH = "weights/efficientad/autoencoder.pth"
TEACHER_PATH = "weights/efficientad/teacher.pth"
STUDENT_PATH = "weights/efficientad/student.pth"
PARAMS_PATH = "weights/efficientad/params.npz"  # Save normalization parameters

# EfficientAD specific transform
efficientad_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Minimum size accepted by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def post_process_map(anomaly_map, sigma=2):
    """Smooth anomaly map with gaussian filter"""
    return gaussian(anomaly_map, sigma=sigma)

class EfficientADModel:
    def __init__(self, threshold=0.6009292970176296):
        self.autoencoder = None
        self.teacher = None
        self.student = None
        self.threshold = threshold
        
        # Normalization parameters
        self.teacher_mean = None
        self.teacher_std = None
        self.q_st0 = None
        self.q_st1 = None
        self.q_ae0 = None
        self.q_ae1 = None
        
        self.load_models()
        self.load_normalization_params()
    
    def load_models(self):
        """Load all model components"""
        try:
            # Load autoencoder model
            if os.path.exists(AUTOENCODER_PATH):
                try:
                    checkpoint = torch.load(AUTOENCODER_PATH, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.autoencoder = torch.load(AUTOENCODER_PATH, map_location=device)
                    else:
                        self.autoencoder = checkpoint
                    
                    self.autoencoder.to(device).eval()
                except Exception as e:
                    print(f"Error loading autoencoder: {str(e)}")
            else:
                print(f"Error: {AUTOENCODER_PATH} file not found!")
            
            # Load teacher model
            if os.path.exists(TEACHER_PATH):
                try:
                    checkpoint = torch.load(TEACHER_PATH, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.teacher = torch.load(TEACHER_PATH, map_location=device)
                    else:
                        self.teacher = checkpoint
                    
                    self.teacher.to(device).eval()
                except Exception as e:
                    print(f"Error loading teacher: {str(e)}")
            else:
                print(f"Error: {TEACHER_PATH} file not found!")
            
            # Load student model
            if os.path.exists(STUDENT_PATH):
                try:
                    checkpoint = torch.load(STUDENT_PATH, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.student = torch.load(STUDENT_PATH, map_location=device)
                    else:
                        self.student = checkpoint
                    
                    self.student.to(device).eval()
                except Exception as e:
                    print(f"Error loading student: {str(e)}")
            else:
                print(f"Error: {STUDENT_PATH} file not found!")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def load_normalization_params(self):
        """Load normalization parameters and check types"""
        try:
            if os.path.exists(PARAMS_PATH):
                params = np.load(PARAMS_PATH)

                # Teacher mean and std as tensors
                if 'teacher_mean' in params:
                    teacher_mean_val = params['teacher_mean']
                    self.teacher_mean = torch.tensor(teacher_mean_val, dtype=torch.float32).to(device)
                else:
                    print("Warning: 'teacher_mean' not found in NPZ file.")
                    
                if 'teacher_std' in params:
                    teacher_std_val = params['teacher_std']
                    self.teacher_std = torch.tensor(teacher_std_val, dtype=torch.float32).to(device)
                else:
                    print("Warning: 'teacher_std' not found in NPZ file.")

                # Scalar normalization parameters
                for q_param_name in ['q_st0', 'q_st1', 'q_ae0', 'q_ae1']:
                    if q_param_name in params:
                        q_val_raw = params[q_param_name]
                        q_val_float = float(q_val_raw.item() if hasattr(q_val_raw, 'item') else q_val_raw)
                        setattr(self, q_param_name, q_val_float)
                    else:
                        print(f"Warning: '{q_param_name}' not found in NPZ file. Default value will be used. {q_param_name} = {getattr(self, q_param_name)}")

                        default_values = {
                            'q_st0': 0.05134373530745506,
                            'q_st1': 1.5087416172027588,
                            'q_ae0': 0.022783687338232994,
                            'q_ae1': 0.3739982843399048
                        }
                        setattr(self, q_param_name, default_values[q_param_name])
            else:
                print(f"Warning: {PARAMS_PATH} not found! Default parameters will be used.")
                self.q_st0 = 0.05134373530745506
                self.q_st1 = 1.5087416172027588
                self.q_ae0 = 0.022783687338232994
                self.q_ae1 = 0.3739982843399048
        except Exception as e:
            print(f"Error loading normalization parameters: {str(e)}")
            self.q_st0 = 0.05134373530745506
            self.q_st1 = 1.5087416172027588
            self.q_ae0 = 0.022783687338232994
            self.q_ae1 = 0.3739982843399048
    
    def predict(self, processed_image):
        """
        Predict anomaly detection using preprocessed image.
        
        Args:
            processed_image: Preprocessed (normalized, background removed) image
            
        Returns:
            dict: Prediction result and anomaly score
        """
        print("EfficientAD predict starting...")
        if any(model is None for model in [self.autoencoder, self.teacher, self.student]):
            print("Error: Some model components not loaded!")
            return {
                "error": "Some model components not loaded. Please check if all weight files are in the correct location.",
                "result": "error"
            }
        
        try:
            # Check image size
            h, w = processed_image.shape[:2]
            print(f"Image size: {h}x{w}")
            if h < 32 or w < 32:
                return {
                    "error": f"Image size too small ({h}x{w}). Minimum 32x32 pixels required.",
                    "result": "error"
                }
                
            # Convert image to tensor format
            try:
                image_tensor = efficientad_transform(processed_image).unsqueeze(0).to(device)
                print(f"Tensor shape: {image_tensor.shape}")
            except Exception as e:
                print(f"Error: Tensor conversion: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "error": f"Error converting image to tensor: {str(e)}",
                    "result": "error"
                }
            
            try:
                combined_map, st_map, ae_map = efficientad_predict(
                    image_tensor, 
                    self.teacher, 
                    self.student, 
                    self.autoencoder,
                    self.teacher_mean, 
                    self.teacher_std, 
                    self.q_st0, 
                    self.q_st1, 
                    self.q_ae0, 
                    self.q_ae1
                )
                
                anomaly_map = combined_map[0, 0].cpu().numpy()
                
                anomaly_map = post_process_map(anomaly_map)
                
                score = float(np.max(anomaly_map))
                
                
            except Exception as e:
                print(f"Error: Predict process: {str(e)}")
                return {
                    "error": f"Error during anomaly detection: {str(e)}",
                    "result": "error"
                }
            
            # Compare threshold
            result = "defect" if score > self.threshold else "good"
            print(f"Result: {result} (Score: {score:.6f}, Threshold: {self.threshold})")
            
            return {
                "result": result,
                "score": score,
                "threshold": self.threshold,
                "message": "Defect" if result == "defect" else "Good"
            }
        except Exception as e:
            print(f"Error: General error: {str(e)}")
            return {
                "error": f"Error during prediction: {str(e)}",
                "result": "error"
            }

def save_normalization_params(teacher_mean, teacher_std, q_st0, q_st1, q_ae0, q_ae1):
    """
    Save normalization parameters.
    
    Args:
        teacher_mean: Teacher features mean
        teacher_std: Teacher features standard deviation
        q_st0: Student-Teacher minimum
        q_st1: Student-Teacher maximum
        q_ae0: Autoencoder minimum
        q_ae1: Autoencoder maximum
    """
    try:
        # Convert tensor values to numpy array
        if torch.is_tensor(teacher_mean):
            teacher_mean = teacher_mean.cpu().numpy()
        if torch.is_tensor(teacher_std):
            teacher_std = teacher_std.cpu().numpy()
            
        # Save parameters
        np.savez(PARAMS_PATH, 
                 teacher_mean=teacher_mean,
                 teacher_std=teacher_std,
                 q_st0=q_st0,
                 q_st1=q_st1,
                 q_ae0=q_ae0,
                 q_ae1=q_ae1)
        
        print(f"Normalization parameters saved: {PARAMS_PATH}")
        return True
    except Exception as e:
        print(f"Error saving normalization parameters: {str(e)}")
        return False

# Global model instance, loaded on app.py import
_model_instance = None

def get_model_instance(threshold=0.6009292970176296):
    global _model_instance  
    if _model_instance is None:
        try:
            _model_instance = EfficientADModel(threshold=threshold)             
        except Exception as e:
            print(f"Error creating model instance: {str(e)}")
            _model_instance = None
            
    return _model_instance

def predict(processed_image):
    """API predict function"""
    model = get_model_instance()
    if model is None:
        return {
            "error": "Model not loaded. Please check if all weight files are in the correct location.",
            "result": "error"
        }
    return model.predict(processed_image)

def efficientad_predict(image_tensor, teacher, student, autoencoder, 
                        teacher_mean, teacher_std, q_st0, q_st1, q_ae0, q_ae1):
    """
    EfficientAD predict function. Standard EfficientAD logic:
    AE, handles the image (current autoencoder.pth compatibility).
    ae_map = MSE(teacher_features, AE_output_features_from_image)
    """
    with torch.no_grad():
        # 1. Get teacher and student features
        teacher_features_raw = teacher(image_tensor) 
        student_features = student(image_tensor)
        
        # 2. Normalize teacher features (used for ST and AE comparison)
        teacher_features_processed = teacher_features_raw
        if teacher_mean is not None and teacher_std is not None:
            teacher_features_processed = (teacher_features_raw - teacher_mean) / teacher_std
        
        # 3. Calculate Student-Teacher difference (MSE) and map
        # Adapt student features to teacher feature size/dimension
        student_features_adapted = student_features[:, :teacher_features_processed.shape[1]]
        st_map_raw = torch.mean(torch.pow(teacher_features_processed - student_features_adapted, 2), dim=1, keepdim=True)
        
        # 4. Normalize and clamp Student-Teacher map
        st_map_norm = (st_map_raw - q_st0) / (q_st1 - q_st0)
        st_map_norm = torch.clamp(st_map_norm, 0, 1)
        
        # 5. Generate teacher-like features from image with Autoencoder
        ae_reconstruction_features = autoencoder(image_tensor) # AE, directly handles image_tensor
        
        # 6. Compare AE output (ae_reconstruction_features) with teacher_features_processed
        # Resize dimensions (especially spatial) if necessary.
        # teacher_features_processed: [B, C, H_t, W_t]
        # ae_reconstruction_features: [B, C, H_ae, W_ae] (C should be the same, e.g. 384)
        
        ae_reconstruction_resized = ae_reconstruction_features
        if ae_reconstruction_features.shape[2:] != teacher_features_processed.shape[2:]:
            ae_reconstruction_resized = F.interpolate(
                ae_reconstruction_features, 
                size=teacher_features_processed.shape[2:], # resize to teacher feature spatial dimensions
                mode='bilinear',
                align_corners=False
            )
        
        # Channel numbers should be the same between teacher_features_processed and ae_reconstruction_resized
        if teacher_features_processed.shape[1] != ae_reconstruction_resized.shape[1]:
            print(f"Warning: Teacher ({teacher_features_processed.shape[1]}) and AE resized ({ae_reconstruction_resized.shape[1]}) feature channels are different!")
            # This is an error, but we can proceed by averaging (not ideal)
            teacher_for_ae_comp = torch.mean(teacher_features_processed, dim=1, keepdim=True)
            ae_for_ae_comp = torch.mean(ae_reconstruction_resized, dim=1, keepdim=True)
        else:
            teacher_for_ae_comp = teacher_features_processed
            ae_for_ae_comp = ae_reconstruction_resized

        ae_map_raw = torch.mean(torch.pow(teacher_for_ae_comp - ae_for_ae_comp, 2), dim=1, keepdim=True)

        # 7. Normalize and clamp Autoencoder map
        # q_ae0 and q_ae1 should be suitable for this calculation method.
        ae_map_norm = (ae_map_raw - q_ae0) / (q_ae1 - q_ae0)
        ae_map_norm = torch.clamp(ae_map_norm, 0, 1)
        
        # 8. Combine two maps
        # st_map_norm and ae_map_norm should now have the same spatial dimensions (H_t, W_t).
        if st_map_norm.shape[2:] != ae_map_norm.shape[2:]:
            # This should no longer happen, for safety check
            print(f"Warning: ST ({st_map_norm.shape}) and AE ({ae_map_norm.shape}) map dimensions are still different! Resizing AE...")
            ae_map_norm = F.interpolate(
                ae_map_norm, 
                size=st_map_norm.shape[2:], 
                mode='bilinear',
                align_corners=False
            )
        
        combined_map = (st_map_norm + ae_map_norm) / 2.0
        combined_map = torch.clamp(combined_map, 0, 1)
        
        return combined_map, st_map_norm, ae_map_norm 