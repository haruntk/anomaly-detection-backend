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
Örnek bir modeli kullanmak için aşağıdaki adımları takip edin:

python app.py

Daha sonra server'a istek atabilirsiniz.

📊 Sonuçlar
Her model için eğitim ve test sonuçları, görselleştirmelerle birlikte colab üzerindedir. Hanüz repoya dahil edilmemiştir.
