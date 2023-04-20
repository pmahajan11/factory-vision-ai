from flask import Flask, request, jsonify
from urllib.request import urlopen
from PIL import Image
import numpy as np
import json
import classifier


app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return "Welcome to Image Classifier!"

@app.route("/image", methods=['POST'])
def predict():
    try:
        image_url = json.loads(request.get_data().decode('utf-8'))['url']
        with urlopen(image_url) as img:
            image = Image.open(img)
        if image.mode != "RGB":
            image = image.convert("RGB")
        print("predicting...")
        image_class, prob = classifier.predict_image(image)
        return jsonify(
            {"image_class": image_class,
             "probability": prob}
        )
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)