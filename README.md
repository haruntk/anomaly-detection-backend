# 🧠 Anomaly Detection Repository

Bu proje, görüntü tabanlı anomali tespiti için çeşitli derin öğrenme tabanlı yöntemleri içermektedir. Kullanılan başlıca modeller:

- ✅ **PaDiM** (Patch Distribution Modeling)
- ⚡ **EfficientAD**
- 🧬 **UniNet**

## 📁 İçerik

Bu repoda farklı anomaly detection modelleriyle yapılmış deneyler, eğitim ve test scriptleri, ve veri ön işleme adımları yer almaktadır.

## 🚀 Kurulum

Projeyi klonladıktan sonra aşağıdaki adımları izleyerek gerekli paketleri kurabilirsiniz:

cd anomaly-detection-backend
pip install -r requirements.txt
requirements.txt dosyasında ihtiyaç duyulan tüm Python kütüphaneleri listelenmiştir.

## 🧪 Kullanılan Modeller

🔹 PaDiM
Özellik çıkarımı için önceden eğitilmiş backbone (örn. ResNet)

Gaussian dağılım tahmini ile anomali tespiti

🔹 EfficientAD
Hafif ve hızlı bir anomaly detection modeli

Düşük hesaplama maliyeti ile yüksek doğruluk

🔹 UniNet
Farklı anomaly detection stratejilerini birleştiren evrensel ağ

Görüntü işleme görevlerinde güçlü genel performans

## 🖼️ Veri Setleri

Proje, MvTec AD veri seti ile uyumludur.

## ⚙️ Kullanım
Örnek bir model çalıştırmak için aşağıdaki adımları takip edin:

bash
Kopyala
Düzenle
python train.py --model padim --dataset mvtec --category bottle
veya test için:

bash
Kopyala
Düzenle
python test.py --model efficientad --dataset visa --category cable
train.py ve test.py scriptleri proje kök dizininde yer almaktadır.

📊 Sonuçlar
Her model için eğitim ve test sonuçları, görselleştirmelerle birlikte results/ klasöründe tutulur. ROC-AUC, PRO ve F1 gibi metriklerle değerlendirme yapılmıştır.## Renk Referansı

| Renk             | Hex                                                                |
| ----------------- | ------------------------------------------------------------------ |
| örnek renk | ![#0a192f](https://via.placeholder.com/10/0a192f?text=+) #0a192f |
| örnek renk | ![#f8f8f8](https://via.placeholder.com/10/f8f8f8?text=+) #f8f8f8 |
| örnek renk | ![#00b48a](https://via.placeholder.com/10/00b48a?text=+) #00b48a |
| örnek renk | ![#00d1a0](https://via.placeholder.com/10/00b48a?text=+) #00d1a0 | 
