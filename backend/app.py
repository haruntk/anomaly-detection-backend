from flask import Flask, request, jsonify
from flask_cors import CORS
from models import padim, efficientad, uninet
from utils.preprocess import load_image, preprocess_image
import os
import traceback
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get("model", "padim")
        
        if 'image' not in request.files:
            return jsonify({"error": "Görüntü dosyası bulunamadı"}), 400
            
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"error": "Dosya seçilmedi"}), 400

        # Görüntüyü yükle
        try:
            image = load_image(image_file)
        except Exception as e:
            return jsonify({"error": f"Görüntü yüklenirken hata oluştu: {str(e)}"}), 400
        
        # Görüntüyü ön işleme adımlarından geçir
        try:
            # Modele göre hedef boyutu ayarla
            target_size = 256
            remove_bg = True # PaDiM için varsayılan
            if model_type == "efficientad":
                target_size = 256  # EfficientAD için test edilen çalışan boyut
                remove_bg = False # EfficientAD için Colab ile uyum
            elif model_type == "uninet":
                target_size = 256  # UniNet için ayarla
                remove_bg = False  # UniNet için arka plan kaldırmaya gerek yok
                
            processed_image = preprocess_image(image, remove_background=remove_bg, target_size=target_size)
            
            # DEBUG BAŞLANGICI - app.py - Gelen dosya adı ve işlenmiş görüntü
            print(f"DEBUG APP: Gelen dosya adı: {image_file.filename}")
            if isinstance(processed_image, np.ndarray):
                print(f"DEBUG APP: processed_image (uninet.predict öncesi) - shape: {processed_image.shape}, min: {processed_image.min():.6f}, max: {processed_image.max():.6f}, mean: {processed_image.mean():.6f}, dtype: {processed_image.dtype}")
            else:
                print(f"DEBUG APP: processed_image (uninet.predict öncesi) beklenen formatta değil: {type(processed_image)}")
            # DEBUG SONU - app.py
        except Exception as e:
            return jsonify({"error": f"Görüntü işlenirken hata oluştu: {str(e)}"}), 400

        # Threshold değerini formdan al
        # Not: Yeni skorlama yöntemi (1/varyans) için varsayılan eşik 1500 olarak ayarlandı
        threshold = float(request.form.get("threshold", 1500.0))

        # Model seçimi ve tahmin
        try:
            if model_type == "padim":
                result = padim.predict(processed_image)
            elif model_type == "efficientad":
                result = efficientad.predict(processed_image)
            elif model_type == "uninet":
                result = uninet.predict(processed_image, threshold)
            else:
                return jsonify({"error": "Model desteklenmiyor"}), 400
        except Exception as e:
            error_msg = f"Tahmin sırasında hata oluştu: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())  # Detaylı hata çıktısı
            return jsonify({"error": error_msg}), 500

        # Hata kontrolü
        if result.get("result") == "error":
            print(f"Model hata döndürdü: {result.get('error', 'Bilinmeyen hata')}")
            return jsonify(result), 500

        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Beklenmeyen bir hata oluştu: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())  # Detaylı hata çıktısı
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    # weights klasörünün varlığını kontrol et
    if not os.path.exists("weights"):
        os.makedirs("weights")
        print("Uyarı: weights klasörü oluşturuldu. Lütfen model ağırlıklarını bu klasöre yerleştirin.")
    
    # UniNet için gerekli klasörleri oluştur
    uninet_path = os.path.join("weights", "UniNet", "wood")
    if not os.path.exists(uninet_path):
        os.makedirs(uninet_path)
        print(f"Uyarı: {uninet_path} klasörü oluşturuldu. UniNet model ağırlıklarını buraya yerleştirin.")
    
    app.run(debug=True)