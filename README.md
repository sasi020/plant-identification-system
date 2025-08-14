# 🌿 Plant Identification using MobileNetV2

## Overview  
This project identifies **40 types of medicinal plants** from images using the **MobileNetV2** deep learning architecture.  
The model is trained on a **10 GB dataset** sourced from [Kaggle](https://www.kaggle.com/), achieving high accuracy in plant classification.  
It is deployed as a **Flask web application** for real-time predictions, allowing users to upload plant images and get instant results.  

---

## Features  
- **Real-Time Plant Identification** – Upload a plant image to instantly identify the species.  
- **Probability Scores** – Displays prediction confidence for better decision-making.  
- **User-Friendly Interface** – Simple web UI built with HTML and CSS.  
- **Lightweight Model** – MobileNetV2 ensures fast and efficient performance.  
- **Custom Dataset Support** – Easily retrain with your own dataset.  

---

## Project Highlights  
- **Pre-Trained Model**:  
  - A trained MobileNetV2 model (`plant_identification_mobilenetv2.h5`) is provided for direct use.  
- **Custom Training**:  
  - Easily train your own model using the provided training script and dataset.  
- **Scalable Deployment**:  
  - Designed to be deployable on local servers or cloud platforms like Heroku, Render, or AWS.  

---

## Tech Stack  
- **Language**: Python  
- **Libraries**: Flask, TensorFlow, NumPy, Pillow  
- **Frontend**: HTML, CSS  
- **Model**: MobileNetV2 (Transfer Learning)  

---

## Usage  

### **Option 1 – Use the Pre-Trained Model**  
1. Clone this repository:
```bash
git clone https://github.com/yourusername/plant-identification-mobilenetv2.git
cd plant-identification-mobilenetv2
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask app:
```bash
python app.py
```
4. Open your browser and go to:
```
http://127.0.0.1:5000/
```

---

### **Option 2 – Train Your Own Model**  
1. Download the dataset from Kaggle.  
2. Modify the training script to point to your dataset location.  
3. Run the training script to generate a new `.h5` model file.  
4. Replace the existing model in the project folder.  

---

## How the Project Was Built  

1. **Dataset Collection**  
   - Dataset: **40 plant species**, 10 GB size.  
   - Sourced from Kaggle.  

2. **Data Preprocessing**  
   - Images resized to 224x224 pixels.  
   - Normalization applied for faster convergence.  

3. **Model Training**  
   - MobileNetV2 used with transfer learning.  
   - Softmax output layer for multi-class classification.  
   - Achieved high accuracy after fine-tuning.  

4. **Deployment**  
   - Flask backend for prediction API.  
   - Simple HTML/CSS frontend for image uploads.  

---

## Future Enhancements  
- Add more plant species.  
- Deploy as a mobile app.  
- Add plant medicinal benefits information in results.  
- Optimize for faster predictions on low-end devices.  

---

## Contributors  
- **Sura Sasi Kumar Reddy**  
- **D. Jhushi**  
- **S. Gopi Chand**  
- **R. M. V. S. Murthy**  

---

## License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
