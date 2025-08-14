import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Step 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Get the list of class folders directly from the dataset directory
num_classes = len([folder for folder in os.listdir('/content/drive/MyDrive/Medicinal plant dataset') if os.path.isdir(os.path.join('/content/drive/MyDrive/Medicinal plant dataset', folder))])
print(f"Number of classes: {num_classes}")

# Load the MobileNetV2 model pre-trained on ImageNet, excluding the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Build the custom model on top of MobileNetV2
model = Sequential([
    base_model,  # Pre-trained feature extractor
    GlobalAveragePooling2D(),  # Reduce dimensions and prevent overfitting
    Dense(256, activation='relu'),  # Fully connected layer with ReLU activation
    Dense(128, activation='relu'),  # Another dense layer
    Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
])


# Compile the model with Adam optimizer and sparse categorical crossentropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing for training and validation datasets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_datagen.flow_from_directory('/content/drive/MyDrive/Medicinal plant dataset',
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='sparse',
                                               subset='training')
validation_data = train_datagen.flow_from_directory('/content/drive/MyDrive/Medicinal plant dataset',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='sparse',
                                                    subset='validation')


# Train the model and store training history
history = model.fit(train_data, validation_data=validation_data, epochs=10)


# Save the trained model
model.save('/content/drive/MyDrive/plant_identification_mobilenetv2.h5')


# Plot training and validation accuracy with epoch markers
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss with epoch markers
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
