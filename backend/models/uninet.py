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

# UniNet modül importları
from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.model import UniNet

# UniNet'in kullandığı utils fonksiyonlarını tanımlayalım
def to_device(all_models, device):
    """Modelleri belirtilen cihaza taşı"""
    to_models = []
    for i in all_models:
        i.to(device)
        to_models.append(i)
    return to_models

def load_weights(modules_list, ckpt_path, suffix):
    """Model ağırlıklarını yükle"""
    print(f"Model ağırlıkları yükleniyor: {os.path.join(ckpt_path, f'{suffix}.pth')}")
    
    # Map location ile cihaz bağımsız yükleme
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        state_dict = torch.load(os.path.join(ckpt_path, f"{suffix}.pth"), map_location=device)
    except Exception as e:
        print(f"Ağırlık dosyası yüklenirken hata: {str(e)}")
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
                print("DFS modülü 'strict=False' ile yükleniyor (eksik anahtarlar yoksayılacak)")
                # DFS anahtarının state_dict içinde olup olmadığını kontrol et
                if 'dfs' in state_dict:
                    print(f"BİLGİ: 'dfs' anahtarı, {os.path.join(ckpt_path, f'{suffix}.pth')} dosyasındaki state_dict içinde bulundu.")
                    module.load_state_dict(state_dict['dfs'], strict=False)
                else:
                    print(f"UYARI: 'dfs' anahtarı, {os.path.join(ckpt_path, f'{suffix}.pth')} dosyasındaki state_dict içinde BULUNAMADI.")
                    print(f"UYARI: Yüklenen ağırlık dosyasındaki kullanılabilir anahtarlar: {list(state_dict.keys())}")
                    print("UYARI: DFS modülü, yüklenen ağırlıklar olmadan (muhtemelen başlangıç/rastgele ağırlıklarla) kullanılacak.")
                    module.load_state_dict({}, strict=False) # Anahtar bulunamazsa boş dict ile devam et
            else:
                module.load_state_dict(state_dict[str(key)])
        except Exception as e:
            print(f"{key} modülü için ağırlıklar yüklenirken hata: {str(e)}")
            raise
            
        module.eval()
        module.to(device)  # Cihaza gönder (CPU veya CUDA)
        new_state[str(key)] = module

    return new_state

# Model ve çıktılar için önbellek
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
    """UniNet modelini yükle"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"UniNet modeli {device} üzerinde çalışacak")
    
    c = Config()
    
    # Model dosyası path
    ckpt_path = os.path.join("weights", "UniNet", "wood")
    model_path = os.path.join(ckpt_path, "BEST_P_PRO.pth")
    
    # Model ağırlıklarının varlığını kontrol et
    if not os.path.exists(model_path):
        print(f"UYARI: Model ağırlıkları {model_path} bulunamadı. Lütfen ağırlık dosyasını ekleyin.")
        print("UniNet model dosyaları bulunamadığından yükleme iptal edildi.")
        return False
    
    try:
        # Modeli oluştur
        print("UniNet model bileşenleri oluşturuluyor...")
        Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
        Source_teacher.layer4 = None
        Source_teacher.fc = None
        
        student = de_wide_resnet50_2(pretrained=False)
        DFS = DomainRelated_Feature_Selection()
        
        # Modülleri cihaza (CPU veya CUDA) gönder
        print(f"Model bileşenleri {device} cihazına aktarılıyor...")
        [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
        Target_teacher = copy.deepcopy(Source_teacher)
        
        # Model ağırlıklarını yükle
        print(f"UniNet modeli yükleniyor: {model_path}")
        
        try:
            # Model ağırlıklarını oku, ancak DFS için hataya hazırlıklı ol
            new_state = load_weights([Target_teacher, bn, student, DFS], ckpt_path, "BEST_P_PRO")
            Target_teacher = new_state['tt']
            bn = new_state['bn']
            student = new_state['st']
            
            # DFS null olabilir, kontrol et
            if new_state['dfs'] is None:
                print("UYARI: DFS modülü için ağırlık yüklenemedi. Varsayılan değerler kullanılacak.")
            else:
                DFS = new_state['dfs']
            
            # Modeli oluştur
            model = UniNet(c, Source_teacher.eval(), Target_teacher, bn, student, DFS)
            model.eval()
            print("UniNet modeli başarıyla yüklendi")
            return True
            
        except Exception as e:
            print(f"Model ağırlıkları yüklenirken hata oluştu: {str(e)}")
            print("Hata ayrıntıları:")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"UniNet modeli başlatılırken hata: {str(e)}")
        return False

def create_heatmap(original_img, anomaly_map, threshold=0.25):
    """Anomali haritasından görselleştirme oluştur"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Orijinal Görüntü')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map, cmap='jet')
    plt.title('Anomali Haritası')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.imshow(anomaly_map, cmap='jet', alpha=0.4)
    plt.title('Anomali Tespiti')
    plt.axis('off')
    
    anomaly_score = float(np.mean(anomaly_map))
    prediction = "Defect (Anomali)" if anomaly_score > threshold else "Good (Normal)"
    
    plt.suptitle(f'Tahmin: {prediction} (Skor: {anomaly_score:.4f}, Eşik: {threshold})', fontsize=16)
    plt.tight_layout()
    
    # Görüntüyü bir BytesIO nesnesine kaydet
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Base64 kodunu al
    heatmap_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return heatmap_str, anomaly_score, prediction

def predict(image, threshold=5.0):
    """UniNet modeli ile tahmin yap"""
    global model, device
    
    # Model henüz yüklenmemişse yükle
    if model is None:
        success = load_model()
        if not success:
            return {"result": "error", "error": "UniNet modeli yüklenemedi"}
    
    try:
        # NumPy array'i Torch tensörüne dönüştür
        # Görüntü zaten preprocess_image ile hazırlanmış olacak
        # Ama UniNet için özel normalizasyon gerekiyorsa burada ekleyin
        if isinstance(image, np.ndarray):
            # NumPy array'i PIL görüntüsüne dönüştür
            # Gelen 'image' (yani app.py'deki processed_image) float32 ve [0,1] aralığında ise
            # önce [0,255] aralığına getirip uint8'e çevirmeliyiz.
            if image.dtype == np.float32 or image.dtype == np.float64:
                if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                    print("DEBUG UNINET: Görüntü float ve [0,1] aralığında. [0,255] uint8'e dönüştürülüyor.")
                    image_uint8 = (image * 255.0).round().astype(np.uint8)
                else: # Beklenmedik float aralığı, olduğu gibi uint8'e çevirmeyi dene
                    print(f"DEBUG UNINET: Görüntü float ama beklenen [0,1] aralığında değil (min: {np.min(image)}, max: {np.max(image)}). Doğrudan uint8'e dönüştürülüyor.")
                    image_uint8 = image.astype(np.uint8)
            elif image.dtype == np.uint8:
                print("DEBUG UNINET: Görüntü zaten uint8.")
                image_uint8 = image
            else:
                print(f"DEBUG UNINET: Bilinmeyen görüntü dtype ({image.dtype}). Doğrudan uint8'e dönüştürülüyor.")
                image_uint8 = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image_uint8)
            
            # UniNet için özel transform
            transform = T.Compose([
                T.Resize((256, 256), InterpolationMode.LANCZOS),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(pil_image).unsqueeze(0).to(device)

            # DEBUG BAŞLANGICI - img_tensor
            print(f"DEBUG: img_tensor - shape: {img_tensor.shape}, min: {img_tensor.min().item():.6f}, max: {img_tensor.max().item():.6f}, mean: {img_tensor.mean().item():.6f}, dtype: {img_tensor.dtype}")
            # DEBUG SONU - img_tensor
        else:
            return {"result": "error", "error": "Geçersiz görüntü formatı"}
        
        # Tahmin yap
        with torch.no_grad():
            model.eval()
            
            try:
                # Test modunda model çıktısını al
                x_s = img_tensor
                model.train_or_eval('eval')
                features = model(x_s)
                
                # COLAB YAKLAŞIMI: Doğrudan model çıktılarının maksimum değerlerini kullan
                if isinstance(features, tuple) and len(features) == 2:
                    score_patches = features[0]  # Bu, features_s_t olacak
                    raw_anomaly_score = 0.0  # Varsayılan skor
                    
                    if isinstance(score_patches, torch.Tensor):
                        # Eğer tekil bir tensör ise, maksimum değerini al
                        raw_anomaly_score = torch.max(score_patches).cpu().item()
                        print(f"DEBUG: raw_anomaly_score (tekil tensör): {raw_anomaly_score:.6f}")
                    elif isinstance(score_patches, list) and len(score_patches) > 0:
                        # Eğer tensör listesi ise, her birinin maksimumunu al ve en büyüğünü seç
                        scores = [torch.max(patch).cpu().item() for patch in score_patches if isinstance(patch, torch.Tensor)]
                        raw_anomaly_score = max(scores) if scores else 0.0
                        print(f"DEBUG: raw_anomaly_score (tensör listesi): {raw_anomaly_score:.6f}, tüm skorlar: {scores}")
                    else:
                        return {"result": "error", "error": "Model çıktısı beklenen formatta değil"}
                    
                    # -- ESKİ YAKLAŞIM (görselleştirme için) --
                    # Özellikler arası farkı hesapla (öğretmen-öğrenci farkı)
                    features_s_t, features_stu = features
                    if isinstance(features_s_t, list) and isinstance(features_stu, list) and len(features_s_t) > 0 and len(features_stu) > 0:
                        # İlk özelliği al
                        teacher_feat = features_s_t[0]
                        student_feat = features_stu[0]

                        # DEBUG BAŞLANGICI - teacher_feat ve student_feat
                        print(f"DEBUG: teacher_feat - shape: {teacher_feat.shape}, min: {teacher_feat.min().item():.6f}, max: {teacher_feat.max().item():.6f}, mean: {teacher_feat.mean().item():.6f}")
                        print(f"DEBUG: student_feat - shape: {student_feat.shape}, min: {student_feat.min().item():.6f}, max: {student_feat.max().item():.6f}, mean: {student_feat.mean().item():.6f}")
                        # DEBUG SONU - teacher_feat ve student_feat
                        
                        # Anomali haritasını hesapla (görselleştirme için)
                        anomaly_map_tensor = torch.sum((student_feat - teacher_feat) ** 2, dim=1, keepdim=True)
                        
                        # DEBUG BAŞLANGICI - anomaly_map_tensor (interpolasyon öncesi)
                        print(f"DEBUG: anomaly_map_tensor (interpolasyon öncesi) - shape: {anomaly_map_tensor.shape}, min: {anomaly_map_tensor.min().item():.6f}, max: {anomaly_map_tensor.max().item():.6f}, mean: {anomaly_map_tensor.mean().item():.6f}")
                        # DEBUG SONU - anomaly_map_tensor (interpolasyon öncesi)

                        # ALTERNATİF SKOR HESAPLAMALARI
                        # 1. Üst %5'lik dilimdeki değerlerin ortalaması (tensör üzerinden)
                        k = int(0.05 * anomaly_map_tensor.numel())  # Üst %5
                        topk_values = torch.topk(anomaly_map_tensor.view(-1), k).values
                        anomaly_score_top5_pct = torch.mean(topk_values).cpu().item()
                        
                        # 2. Özellik farkının varyansı (değişkenlik ölçüsü)
                        anomaly_score_variance = torch.var(anomaly_map_tensor.view(-1)).cpu().item()
                        # Varyansın tersini al (1/x) - böylece defect görüntüler daha yüksek değer alır
                        anomaly_score_inverse_variance = 10000000.0 / (anomaly_score_variance + 1.0)  # 10^7 ile çarpma sayıların okunabilirliği için
                        
                        # 3. En büyük değer ile ortalama değer arasındaki fark
                        anomaly_score_peak_mean_diff = (torch.max(anomaly_map_tensor) - torch.mean(anomaly_map_tensor)).cpu().item()
                        
                        # 4. Öğretmen-öğrenci farklarının toplamı (tüm özellik seviyeleri)
                        diff_features = []
                        for i in range(min(len(features_s_t), len(features_stu))):
                            if isinstance(features_s_t[i], torch.Tensor) and isinstance(features_stu[i], torch.Tensor):
                                # Her bir özellik seviyesindeki farkı hesapla
                                diff = torch.sum((features_s_t[i] - features_stu[i]) ** 2, dim=1, keepdim=True)
                                # Maksimum değeri al
                                diff_max = torch.max(diff).cpu().item()
                                diff_features.append(diff_max)
                        anomaly_score_diff_sum = sum(diff_features)
                        
                        # DEBUG: ALTERNATİF SKORLARI GÖSTER
                        print(f"DEBUG: ALTERNATİF SKORLAR:")
                        print(f"1. Üst %5 ortalama: {anomaly_score_top5_pct:.6f}")
                        print(f"2. Varyans: {anomaly_score_variance:.6f}")
                        print(f"2a. 1/Varyans (10^7): {anomaly_score_inverse_variance:.6f}")
                        print(f"3. Peak-Mean farkı: {anomaly_score_peak_mean_diff:.6f}")
                        print(f"4. Farkların toplamı: {anomaly_score_diff_sum:.6f}")
                        print(f"5. Ham maksimum (mevcut): {raw_anomaly_score:.6f}")

                        # Boyutlandır (görselleştirme için)
                        anomaly_map = F.interpolate(anomaly_map_tensor, size=(256, 256), mode='bilinear', align_corners=False)
                        anomaly_map = anomaly_map.squeeze().cpu().numpy()
                        
                        # DEBUG BAŞLANGICI
                        print(f"DEBUG: anomaly_map (normalizasyon öncesi) min: {np.min(anomaly_map)}, max: {np.max(anomaly_map)}, mean: {np.mean(anomaly_map)}")
                        # DEBUG SONU

                        # Min-Max normalizasyon (sadece görselleştirme için)
                        if np.max(anomaly_map) > np.min(anomaly_map):
                            anomaly_map_normalized = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
                            
                            # 5. Normalize edilmiş haritanın üst %20'sindeki değerlerin ortalaması
                            threshold_percentile = 80  # Üst %20
                            mask = anomaly_map_normalized > np.percentile(anomaly_map_normalized, threshold_percentile)
                            
                            if np.sum(mask) > 0:  # Eğer maske boş değilse
                                anomaly_score_top20_norm = float(np.mean(anomaly_map_normalized[mask]))
                            else:
                                anomaly_score_top20_norm = float(np.mean(anomaly_map_normalized))
                                
                            # DEBUG
                            print(f"6. Norm üst %20 ortalama: {anomaly_score_top20_norm:.6f}")
                            
                            # DEBUG BAŞLANGICI
                            print(f"DEBUG: anomaly_map (normalizasyon sonrası) min: {np.min(anomaly_map_normalized)}, max: {np.max(anomaly_map_normalized)}, mean: {np.mean(anomaly_map_normalized)}")
                            # DEBUG SONU
                        else:
                            # DEBUG BAŞLANGICI
                            print(f"DEBUG: anomaly_map normalizasyonu atlandı (max <= min). min: {np.min(anomaly_map)}, max: {np.max(anomaly_map)}")
                            if not np.any(anomaly_map) and np.all(anomaly_map == 0): # Check if all elements are zero
                                print("DEBUG: anomaly_map (normalizasyon öncesi) tüm elemanlar sıfır.")
                            elif np.all(anomaly_map == anomaly_map.flat[0]): # Check if all elements are the same
                                print(f"DEBUG: anomaly_map (normalizasyon öncesi) tüm elemanlar aynı: {anomaly_map.flat[0]}")
                            # DEBUG SONU
                            anomaly_score_top20_norm = 0.0
                        
                        # EN İYİ AYRIM İÇİN SKOR SEÇİMİ
                        # Test sonuçlarına göre: Maksimum-Ortalama farkı (peak-mean difference) en iyi ayrımı sağlar gibi görünüyor
                        # anomaly_score = raw_anomaly_score  # Mevcut (Colab) yöntem
                        # anomaly_score = anomaly_score_top5_pct  # Alternatif 1
                        # anomaly_score = anomaly_score_variance  # Alternatif 2 - Varyans, normal görüntüler için daha yüksek
                        anomaly_score = anomaly_score_inverse_variance  # Alternatif 2a - 1/Varyans, defect görüntüler için daha yüksek olmalı
                        # anomaly_score = anomaly_score_peak_mean_diff  # Alternatif 3 - Peak-Mean normal görüntüler için daha yüksek çıkıyor
                        # anomaly_score = anomaly_score_diff_sum  # Alternatif 4
                        # anomaly_score = anomaly_score_top20_norm * 100  # Alternatif 5 (skalayı ayarlamak için 100 ile çarpılmış)
                        
                        # DEBUG
                        print(f"DEBUG: Final anomaly_score (seçilen): {anomaly_score}")
                        
                        # Not: Bu skora göre threshold değeri değişecektir
                        # 1/Varyans için threshold yaklaşık 1500 civarında olabilir
                        is_anomaly = anomaly_score > threshold
                        print(f"KARAR: anomaly_score: {anomaly_score}, threshold: {threshold}, is_anomaly: {is_anomaly}")
                        
                        # Heatmap oluştur (görselleştirme için)
                        heatmap_base64, _, _ = create_heatmap(np.array(pil_image.resize((256, 256))), anomaly_map, 0.5)  # Görselleştirme için sabit bir threshold
                        
                        # Sonuç döndür
                        result_str = "defect" if is_anomaly else "good"
                        return {
                            "result": result_str,
                            "model": "uninet", 
                            "score": float(anomaly_score),
                            "threshold": float(threshold),
                            "message": "Kusurlu" if result_str == "defect" else "Sağlam",
                            "heatmap": f"data:image/png;base64,{heatmap_base64}"
                        }
                    else:
                        return {"result": "error", "error": "Model çıktısı boş veya uygun olmayan bir formatta"}
                else:
                    return {"result": "error", "error": "Model çıktısı beklenen formatta değil"}
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"result": "error", "error": f"Tahmin sırasında hata: {str(e)}"}
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"result": "error", "error": f"UniNet ile tahmin yapılırken hata: {str(e)}"}