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
            return jsonify({"error": "Image file not found"}), 400
            
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"error": "File not selected"}), 400

        # Load image
        try:
            image = load_image(image_file)
        except Exception as e:
            return jsonify({"error": f"Error loading image: {str(e)}"}), 400
        
        # Pass image through preprocessing steps
        try:
            target_size = 256
            remove_bg = True
            if model_type == "efficientad":
                target_size = 256
                remove_bg = True
            elif model_type == "uninet":
                target_size = 256
                remove_bg = False
                
            processed_image = preprocess_image(image, remove_background=remove_bg, target_size=target_size)

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400

        # Model selection and prediction
        try:
            if model_type == "padim":
                result = padim.predict(processed_image)
            elif model_type == "efficientad":
                result = efficientad.predict(processed_image)
            elif model_type == "uninet":
                result = uninet.predict(processed_image)
            else:
                return jsonify({"error": "Model not supported"}), 400
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            return jsonify({"error": error_msg}), 500

        if result.get("result") == "error":
            print(f"Model returned error: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500

        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)