# models/uninet.py
import torch
import copy
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import base64
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# UniNet module imports
from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.model import UniNet

def to_device(all_models, device):
    """Send models to device"""
    to_models = []
    for i in all_models:
        i.to(device)
        to_models.append(i)
    return to_models

def load_weights(modules_list, ckpt_path, suffix):
    """Load weights"""
    print(f"Loading weights: {os.path.join(ckpt_path, f'{suffix}.pth')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        state_dict = torch.load(os.path.join(ckpt_path, f"{suffix}.pth"), map_location=device)
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        raise
    
    new_state = {"tt": None,
                "bn": None,
                "st": None,
                "dfs": None
                }
    
    for (module, (key, value)) in zip(modules_list, new_state.items()):
        if module is None:
            continue
        
        try:
            if key == 'dfs':
                if 'dfs' in state_dict:
                    module.load_state_dict(state_dict['dfs'], strict=False)
                else:
                    module.load_state_dict({}, strict=False)    
            else:
                module.load_state_dict(state_dict[str(key)])
        except Exception as e:
            raise
            
        module.eval()
        module.to(device)
        new_state[str(key)] = module

    return new_state

model = None
device = None

class Config:
    def __init__(self):
        self.dataset = "MVTec AD"
        self._class_ = "wood"
        self.setting = "oc"
        self.weighted_decision_mechanism = True
        self.default = 0.3
        self.alpha = 0.01
        self.beta = 0.00003
        self.T = 2.0
        self.domain = "industrial"
        self.image_size = 256
        self.center_crop = 256

def load_model():
    """Load UniNet model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    c = Config()
    
    # Weight file path
    ckpt_path = os.path.join("weights", "UniNet", "wood")
    model_path = os.path.join(ckpt_path, "BEST_P_PRO.pth")
    
    # Check if weights exist
    if not os.path.exists(model_path):
        print(f"Warning: Model weights {model_path} not found. Please add the weight file.")
        return False
    
    try:
        # Create model
        Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
        Source_teacher.layer4 = None
        Source_teacher.fc = None
        
        student = de_wide_resnet50_2(pretrained=False)
        DFS = DomainRelated_Feature_Selection()
        
        # Send models to device (CPU or CUDA)
        [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
        Target_teacher = copy.deepcopy(Source_teacher)
        
        # Load weights
        try:
            # Load weights
            new_state = load_weights([Target_teacher, bn, student, DFS], ckpt_path, "BEST_P_PRO")
            Target_teacher = new_state['tt']
            bn = new_state['bn']
            student = new_state['st']
            
            if new_state['dfs'] is None:
                print("Warning: DFS module weights not loaded. Default values will be used.")
            else:
                DFS = new_state['dfs']   

            # Create model
            model = UniNet(c, Source_teacher.eval(), Target_teacher, bn, student, DFS)
            model.eval()
            print("UniNet model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error loading UniNet model: {str(e)}")
        return False


def predict(image):
    """Predict with UniNet model"""
    global model, device
    
    if model is None:
        success = load_model()
        if not success:
            return {"result": "error", "error": "UniNet model not loaded"}
    
    try:
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                    image_uint8 = (image * 255.0).round().astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
            elif image.dtype == np.uint8:
                image_uint8 = image
            else:
                image_uint8 = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image_uint8)
            
            transform = T.Compose([
                T.Resize((256, 256), InterpolationMode.LANCZOS),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(pil_image).unsqueeze(0).to(device)
        else:
            return {"result": "error", "error": "Invalid image format"}
        
        # Predict
        with torch.no_grad():
            model.eval()
            
            try:
                x_s = img_tensor
                model.train_or_eval('eval')
                features = model(x_s)
                
                if isinstance(features, tuple) and len(features) == 2:
                    features_s_t, features_stu = features
                    if isinstance(features_s_t, list) and isinstance(features_stu, list) and len(features_s_t) > 0 and len(features_stu) > 0:
                        teacher_feat = features_s_t[0]
                        student_feat = features_stu[0]
                        # Calculate anomaly map
                        anomaly_map_tensor = torch.sum((student_feat - teacher_feat) ** 2, dim=1, keepdim=True)
                        # Variance of feature difference
                        anomaly_score_variance = torch.var(anomaly_map_tensor.view(-1)).cpu().item()
                        # Inverse of variance calculation
                        original_score = 10000000.0 / (anomaly_score_variance + 1.0)
                        
                        # Min-max normalization (based on empirical values)
                        min_score = 500  # Minimum expected score
                        max_score = 3000  # Maximum expected score
                        anomaly_score = (original_score - min_score) / (max_score - min_score)
                        # Clip values to ensure they stay in [0,1]
                        anomaly_score = max(0.0, min(1.0, anomaly_score))
                        
                        # Normalize the threshold (1500) using the same parameters
                        threshold = (1500 - min_score) / (max_score - min_score)
                        
                        # Resize (for visualization)
                        anomaly_map = F.interpolate(anomaly_map_tensor, size=(256, 256), mode='bilinear', align_corners=False)
                        anomaly_map = anomaly_map.squeeze().cpu().numpy()

                        is_anomaly = anomaly_score > threshold
                        
                        # Return result
                        result_str = "defect" if is_anomaly else "good"
                        return {
                            "result": result_str,
                            "model": "uninet", 
                            "score": float(anomaly_score),
                            "threshold": float(threshold),
                            "message": "Defect" if result_str == "defect" else "Good",
                        }
                    else:
                        return {"result": "error", "error": "Model output is empty or in an invalid format"}
                else:
                    return {"result": "error", "error": "Model output is not in the expected format"}
                
            except Exception as e:
                return {"result": "error", "error": f"Error during prediction: {str(e)}"}
                
    except Exception as e:
        return {"result": "error", "error": f"Error during UniNet prediction: {str(e)}"}