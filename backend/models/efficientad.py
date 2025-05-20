import torch
import numpy as np
import os
import torch.nn.functional as F
from torchvision import models, transforms
from utils.preprocess import transform_image, extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scipy yüklü mü kontrol et
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Uyarı: scipy.ndimage bulunamadı. Alternatif filtreleme kullanılacak.")

# Skimage yüklü mü kontrol et
try:
    from skimage.filters import gaussian
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Uyarı: skimage.filters bulunamadı. Gaussian için scipy kullanılacak.")

# Model ağırlıklarının yolları
AUTOENCODER_PATH = "weights/efficientad/autoencoder.pth"
TEACHER_PATH = "weights/efficientad/teacher.pth"
STUDENT_PATH = "weights/efficientad/student.pth"
PARAMS_PATH = "weights/efficientad/params.npz"  # Normalizasyon parametrelerini saklama

# EfficientAD için özel transform
efficientad_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Modelin kabul ettiği minimum boyut
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalizasyonu
])

def post_process_map(anomaly_map, sigma=2):
    """Anomali haritasını gaussian filtre ile düzleştirir"""
    if HAS_SKIMAGE:
        # Skimage ile Gaussian filtre (Colab ile aynı)
        print(f"Skimage gaussian uygulanıyor (sigma={sigma})...")
        return gaussian(anomaly_map, sigma=sigma)
    elif HAS_SCIPY:
        print(f"Scipy gaussian uygulanıyor (sigma={sigma})...")
        return gaussian_filter(anomaly_map, sigma=sigma)
    else:
        print("UYARI: Gaussian filtre bulunamadı, filtreleme yapılmıyor!")
        return anomaly_map

class EfficientADModel:
    def __init__(self, threshold=0.6009292970176296):
        self.autoencoder = None
        self.teacher = None
        self.student = None
        self.threshold = threshold
        
        # Normalizasyon parametreleri
        self.teacher_mean = None
        self.teacher_std = None
        self.q_st0 = None
        self.q_st1 = None
        self.q_ae0 = None
        self.q_ae1 = None
        
        self.load_models()
        self.load_normalization_params()
    
    def load_models(self):
        """Tüm model bileşenlerini yükler"""
        try:
            # Autoencoder modelini yükle
            if os.path.exists(AUTOENCODER_PATH):
                try:
                    # Modeli güvenli bir şekilde yükle
                    checkpoint = torch.load(AUTOENCODER_PATH, map_location=device)
                    # Eğer state_dict ise
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.autoencoder = torch.load(AUTOENCODER_PATH, map_location=device)
                    # Doğrudan model ise
                    else:
                        self.autoencoder = checkpoint
                    
                    self.autoencoder.to(device).eval()
                except Exception as e:
                    print(f"Autoencoder yüklenirken hata: {str(e)}")
            else:
                print(f"HATA: {AUTOENCODER_PATH} dosyası bulunamadı!")
            
            # Teacher modelini yükle
            if os.path.exists(TEACHER_PATH):
                try:
                    # Modeli güvenli bir şekilde yükle
                    checkpoint = torch.load(TEACHER_PATH, map_location=device)
                    # Eğer state_dict ise
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.teacher = torch.load(TEACHER_PATH, map_location=device)
                    # Doğrudan model ise
                    else:
                        self.teacher = checkpoint
                    
                    self.teacher.to(device).eval()
                except Exception as e:
                    print(f"Teacher yüklenirken hata: {str(e)}")
            else:
                print(f"HATA: {TEACHER_PATH} dosyası bulunamadı!")
            
            # Student modelini yükle
            if os.path.exists(STUDENT_PATH):
                try:
                    # Modeli güvenli bir şekilde yükle
                    checkpoint = torch.load(STUDENT_PATH, map_location=device)
                    # Eğer state_dict ise
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.student = torch.load(STUDENT_PATH, map_location=device)
                    # Doğrudan model ise
                    else:
                        self.student = checkpoint
                    
                    self.student.to(device).eval()
                except Exception as e:
                    print(f"Student yüklenirken hata: {str(e)}")
            else:
                print(f"HATA: {STUDENT_PATH} dosyası bulunamadı!")
                
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
    
    def load_normalization_params(self):
        """Normalizasyon parametrelerini yükler ve tiplerini kontrol eder."""
        try:
            if os.path.exists(PARAMS_PATH):
                print(f"Normalizasyon parametreleri yükleniyor: {PARAMS_PATH}")
                params = np.load(PARAMS_PATH)
                
                param_keys = list(params.keys())
                print(f"NPZ dosyasındaki anahtarlar: {param_keys}")

                # Teacher mean ve std tensör olarak
                if 'teacher_mean' in params:
                    teacher_mean_val = params['teacher_mean']
                    self.teacher_mean = torch.tensor(teacher_mean_val, dtype=torch.float32).to(device)
                    print(f"  teacher_mean yüklendi, şekil: {self.teacher_mean.shape}, tip: {self.teacher_mean.dtype}, cihaz: {self.teacher_mean.device}")
                    # print(f"  teacher_mean değeri (ilk 5): {self.teacher_mean.flatten()[:5]}")
                else:
                    print("UYARI: NPZ dosyasında 'teacher_mean' bulunamadı.")
                    
                if 'teacher_std' in params:
                    teacher_std_val = params['teacher_std']
                    self.teacher_std = torch.tensor(teacher_std_val, dtype=torch.float32).to(device)
                    print(f"  teacher_std yüklendi, şekil: {self.teacher_std.shape}, tip: {self.teacher_std.dtype}, cihaz: {self.teacher_std.device}")
                    # print(f"  teacher_std değeri (ilk 5): {self.teacher_std.flatten()[:5]}")
                else:
                    print("UYARI: NPZ dosyasında 'teacher_std' bulunamadı.")

                # Skalar normalizasyon parametreleri (doğrudan float olmalı)
                for q_param_name in ['q_st0', 'q_st1', 'q_ae0', 'q_ae1']:
                    if q_param_name in params:
                        q_val_raw = params[q_param_name]
                        # Eğer np array ise .item() ile Python float'a çevir, değilse zaten float'tır
                        # Colab'da .item() ile kaydedilmiş olmalı.
                        q_val_float = float(q_val_raw.item() if hasattr(q_val_raw, 'item') else q_val_raw)
                        setattr(self, q_param_name, q_val_float)
                        print(f"  {q_param_name} yüklendi, değer: {q_val_float}, tip: {type(q_val_float)}")
                    else:
                        print(f"UYARI: NPZ dosyasında '{q_param_name}' bulunamadı. Varsayılan kullanılacak.")
                        # İlgili varsayılan değeri ata (önceki kodunuzdaki gibi)
                        default_values = {
                            'q_st0': 0.05134373530745506,
                            'q_st1': 1.5087416172027588,
                            'q_ae0': 0.022783687338232994,
                            'q_ae1': 0.3739982843399048
                        }
                        setattr(self, q_param_name, default_values[q_param_name])
                        print(f"  {q_param_name} varsayılan olarak ayarlandı: {getattr(self, q_param_name)}")

                print("Normalizasyon parametreleri yüklendi ve atandı (Detaylı Log):")
                print(f"  self.q_st0: {self.q_st0} (tip: {type(self.q_st0)})")
                print(f"  self.q_st1: {self.q_st1} (tip: {type(self.q_st1)})")
                print(f"  self.q_ae0: {self.q_ae0} (tip: {type(self.q_ae0)})")
                print(f"  self.q_ae1: {self.q_ae1} (tip: {type(self.q_ae1)})")
            else:
                print(f"UYARI: {PARAMS_PATH} bulunamadı! Varsayılan parametreler kullanılacak.")
                # Varsayılan parametreler - Colab'dan elde edilen değerler
                self.q_st0 = 0.05134373530745506
                self.q_st1 = 1.5087416172027588
                self.q_ae0 = 0.022783687338232994
                self.q_ae1 = 0.3739982843399048
                print(f"Varsayılan değerler: q_st0={self.q_st0}, q_st1={self.q_st1}, q_ae0={self.q_ae0}, q_ae1={self.q_ae1}")
        except Exception as e:
            print(f"Normalizasyon parametreleri yüklenirken HATA: {str(e)}")
            import traceback
            traceback.print_exc()
            print("HATA nedeniyle varsayılan parametreler kullanılıyor.")
            self.q_st0 = 0.05134373530745506
            self.q_st1 = 1.5087416172027588
            self.q_ae0 = 0.022783687338232994
            self.q_ae1 = 0.3739982843399048
    
    def predict(self, processed_image):
        """
        Ön işlenmiş görüntüyü kullanarak anomali tespiti yapar.
        
        Args:
            processed_image: Ön işlenmiş (normalize edilmiş, arka planı kaldırılmış) görüntü
            
        Returns:
            dict: Tespit sonucu ve anomali skoru
        """
        print("EfficientAD predict başlatılıyor...")
        if any(model is None for model in [self.autoencoder, self.teacher, self.student]):
            print("HATA: Bazı model bileşenleri yüklenemedi!")
            return {
                "error": "Bazı model bileşenleri yüklenemedi. Lütfen tüm ağırlık dosyalarının doğru konumda olduğunu kontrol edin.",
                "result": "error"
            }
        
        try:
            # Görüntü boyutunu kontrol et
            h, w = processed_image.shape[:2]
            print(f"Görüntü boyutu: {h}x{w}")
            if h < 32 or w < 32:
                return {
                    "error": f"Görüntü boyutu çok küçük ({h}x{w}). En az 32x32 piksel olmalıdır.",
                    "result": "error"
                }
                
            # Görüntüyü tensor formatına dönüştür
            try:
                print("Görüntüyü tensor formatına dönüştürme...")
                image_tensor = efficientad_transform(processed_image).unsqueeze(0).to(device)
                print(f"Tensor şekli: {image_tensor.shape}")
            except Exception as e:
                print(f"HATA: Tensor dönüştürme: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "error": f"Görüntü tensor'a dönüştürülürken hata oluştu: {str(e)}",
                    "result": "error"
                }
            
            # Colab'deki predict fonksiyonunu çağır
            try:
                # Normalizasyon parametrelerini yazdır
                print("Normalizasyon parametreleri:")
                print(f"q_st0: {self.q_st0}, q_st1: {self.q_st1}")
                print(f"q_ae0: {self.q_ae0}, q_ae1: {self.q_ae1}")
                
                # Orijinal predict fonksiyonunu kullan
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
                
                # Anomali haritasını CPU'ya taşı ve numpy'a dönüştür
                anomaly_map = combined_map[0, 0].cpu().numpy()
                
                # Post-processing uygula
                anomaly_map = post_process_map(anomaly_map)
                
                # Maksimum değeri skor olarak kullan - Colab ile aynı yaklaşım
                score = float(np.max(anomaly_map))
                print(f"Anomali skoru: {score:.6f}")
                
                # Debug bilgisi için teacher-student ve autoencoder skorlarını da hesapla
                ts_score = float(torch.max(st_map).item())
                ae_score = float(torch.max(ae_map).item())
                print(f"TS skoru: {ts_score:.6f}, AE skoru: {ae_score:.6f}")
                
            except Exception as e:
                print(f"HATA: Predict işlemi: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "error": f"Anomali tespiti sırasında hata oluştu: {str(e)}",
                    "result": "error"
                }
            
            # Eşik değeri ile karşılaştır
            result = "defect" if score > self.threshold else "good"
            print(f"Sonuç: {result} (Skor: {score:.6f}, Eşik: {self.threshold})")
            
            return {
                "result": result,
                "score": score,
                "threshold": self.threshold,
                "ts_score": ts_score,
                "ae_score": ae_score,
                "message": "Kusurlu" if result == "defect" else "Sağlam"
            }
        except Exception as e:
            print(f"HATA: Genel hata: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Tahmin sırasında hata oluştu: {str(e)}",
                "result": "error"
            }

def save_normalization_params(teacher_mean, teacher_std, q_st0, q_st1, q_ae0, q_ae1):
    """
    Normalizasyon parametrelerini kaydetmek için kullanılır.
    
    Args:
        teacher_mean: Teacher özelliklerinin ortalaması
        teacher_std: Teacher özelliklerinin standart sapması
        q_st0: Student-Teacher minimumu
        q_st1: Student-Teacher maksimumu
        q_ae0: Autoencoder minimumu
        q_ae1: Autoencoder maksimumu
    """
    try:
        # Tensör değerleri numpy dizisine dönüştür
        if torch.is_tensor(teacher_mean):
            teacher_mean = teacher_mean.cpu().numpy()
        if torch.is_tensor(teacher_std):
            teacher_std = teacher_std.cpu().numpy()
            
        # Parametreleri kaydet
        np.savez(PARAMS_PATH, 
                 teacher_mean=teacher_mean,
                 teacher_std=teacher_std,
                 q_st0=q_st0,
                 q_st1=q_st1,
                 q_ae0=q_ae0,
                 q_ae1=q_ae1)
        
        print(f"Normalizasyon parametreleri kaydedildi: {PARAMS_PATH}")
        return True
    except Exception as e:
        print(f"Normalizasyon parametreleri kaydedilirken hata oluştu: {str(e)}")
        return False

# Global model örneği, app.py'nin import anında yüklenmesi için
_model_instance = None

def get_model_instance(threshold=0.6009292970176296):
    global _model_instance
    if _model_instance is None:
        try:
            _model_instance = EfficientADModel(threshold=threshold)
            # Normalizasyon parametreleri yoksa ve Colab'den almak için bir yol
            if not os.path.exists(PARAMS_PATH) and _model_instance.teacher_mean is None: # teacher_mean kontrolü ekledik
                print("\n" + "="*80)
                print("ÖNEMLİ: Colab'den normalizasyon parametrelerini almanız önerilir:")
                print("""
# Colab'de çalıştırılacak kod:
import numpy as np
np.savez('params.npz', 
         teacher_mean=teacher_mean.cpu().numpy() if hasattr(teacher_mean, 'cpu') else teacher_mean,
         teacher_std=teacher_std.cpu().numpy() if hasattr(teacher_std, 'cpu') else teacher_std,
         q_st0=q_st0 if isinstance(q_st0, (float, int)) else q_st0.item(),
         q_st1=q_st1 if isinstance(q_st1, (float, int)) else q_st1.item(),
         q_ae0=q_ae0 if isinstance(q_ae0, (float, int)) else q_ae0.item(),
         q_ae1=q_ae1 if isinstance(q_ae1, (float, int)) else q_ae1.item())
                """)
                print("Bu dosyayı 'weights/efficientad/params.npz' konumuna yerleştirin.")
                print("="*80 + "\n")
        except Exception as e:
            print(f"Model oluşturulurken hata oluştu: {str(e)}")
            _model_instance = None # Hata durumunda None olarak kalsın
    return _model_instance

def predict(processed_image):
    """API için predict fonksiyonu"""
    model = get_model_instance()
    if model is None:
        return {
            "error": "Model yüklenemedi. Lütfen model dosyalarınızı kontrol edin.",
            "result": "error"
        }
    return model.predict(processed_image)

# EfficientAD predict işlevi.
# Autoencoder, HAM GÖRÜNTÜYÜ işler (mevcut autoencoder.pth uyumluluğu için).
# Bu, q_ae değerlerinin geçerliliği konusunda soru işareti yaratır.
def efficientad_predict(image_tensor, teacher, student, autoencoder, 
                        teacher_mean, teacher_std, q_st0, q_st1, q_ae0, q_ae1):
    """
    EfficientAD predict işlevi. Standard EFficientAD mantığına göre:
    AE, görüntüyü alır ve öğretmen özelliklerini yeniden oluşturmaya çalışır.
    ae_map = MSE(teacher_features, AE_output_features_from_image)
    """
    with torch.no_grad():
        # 1. Teacher ve Student özelliklerini al
        teacher_features_raw = teacher(image_tensor) 
        student_features = student(image_tensor)
        
        # 2. Teacher özelliklerini normalleştir (ST ve AE karşılaştırması için kullanılacak)
        teacher_features_processed = teacher_features_raw
        if teacher_mean is not None and teacher_std is not None:
            teacher_features_processed = (teacher_features_raw - teacher_mean) / teacher_std
        
        # 3. Student-Teacher farkını (MSE) ve haritasını hesapla
        # Student özelliklerini teacher özellik sayısına/boyutuna uyarla
        student_features_adapted = student_features[:, :teacher_features_processed.shape[1]]
        st_map_raw = torch.mean(torch.pow(teacher_features_processed - student_features_adapted, 2), dim=1, keepdim=True)
        
        # 4. Student-Teacher haritasını normalleştir ve clamp et
        st_map_norm = (st_map_raw - q_st0) / (q_st1 - q_st0)
        st_map_norm = torch.clamp(st_map_norm, 0, 1)
        
        # 5. Autoencoder ile görüntüden öğretmen benzeri özellikler üret
        ae_reconstruction_features = autoencoder(image_tensor) # AE, doğrudan image_tensor alır
        
        # 6. AE çıktısını (ae_reconstruction_features) teacher_features_processed ile karşılaştır.
        # Boyutları (özellikle uzamsal) eşitlemek gerekebilir.
        # teacher_features_processed: [B, C, H_t, W_t]
        # ae_reconstruction_features: [B, C, H_ae, W_ae] (C aynı olmalı, örn. 384)
        
        ae_reconstruction_resized = ae_reconstruction_features
        if ae_reconstruction_features.shape[2:] != teacher_features_processed.shape[2:]:
            # print(f"AE özellik haritası yeniden boyutlandırılıyor: {ae_reconstruction_features.shape[2:]} -> {teacher_features_processed.shape[2:]}")
            ae_reconstruction_resized = F.interpolate(
                ae_reconstruction_features, 
                size=teacher_features_processed.shape[2:], # teacher özelliklerinin uzamsal boyutuna getir
                mode='bilinear',
                align_corners=False
            )
        
        # Kanal sayıları teacher_features_processed ve ae_reconstruction_resized arasında aynı olmalı.
        if teacher_features_processed.shape[1] != ae_reconstruction_resized.shape[1]:
            print(f"UYARI: Teacher ({teacher_features_processed.shape[1]}) ve AE yeniden yapılandırma ({ae_reconstruction_resized.shape[1]}) özellik kanalları farklı!")
            # Bu durumda bir hata var demektir, ancak devam etmek için ortalama alabiliriz (ideal değil)
            teacher_for_ae_comp = torch.mean(teacher_features_processed, dim=1, keepdim=True)
            ae_for_ae_comp = torch.mean(ae_reconstruction_resized, dim=1, keepdim=True)
        else:
            teacher_for_ae_comp = teacher_features_processed
            ae_for_ae_comp = ae_reconstruction_resized

        ae_map_raw = torch.mean(torch.pow(teacher_for_ae_comp - ae_for_ae_comp, 2), dim=1, keepdim=True)

        # 7. Autoencoder haritasını normalleştir ve clamp et
        # q_ae0 ve q_ae1 bu hesaplama yöntemine uygun olmalı.
        ae_map_norm = (ae_map_raw - q_ae0) / (q_ae1 - q_ae0)
        ae_map_norm = torch.clamp(ae_map_norm, 0, 1)
        
        # 8. İki haritayı birleştir
        # st_map_norm ve ae_map_norm artık aynı uzamsal boyutta olmalı (H_t, W_t).
        if st_map_norm.shape[2:] != ae_map_norm.shape[2:]:
             # Bu artık olmamalı, güvenlik için kontrol
            print(f"UYARI: ST ({st_map_norm.shape}) ve AE ({ae_map_norm.shape}) harita boyutları hala farklı! AE yeniden boyutlandırılıyor...")
            ae_map_norm = F.interpolate(
                ae_map_norm, 
                size=st_map_norm.shape[2:], 
                mode='bilinear',
                align_corners=False
            )
        
        combined_map = (st_map_norm + ae_map_norm) / 2.0
        combined_map = torch.clamp(combined_map, 0, 1)
        
        return combined_map, st_map_norm, ae_map_norm 