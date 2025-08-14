from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('D:/project/MobileNetV2/plant_identification_mobilenetv2.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'plant_image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['plant_image']
    if file.filename == '':
        return 'No file selected', 400

    img_path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    with open('D:/project/MobileNetV2/class_labels.txt', 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]

    plant_name = class_labels[predicted_class]
    return render_template('index.html', plant_name=plant_name)

if __name__ == '__main__':
    app.run(debug=True)
