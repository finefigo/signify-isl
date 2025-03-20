import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse
import tensorflowjs as tfjs

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train an ISL gesture recognition model")
parser.add_argument("--data-dir", type=str, default="training_data", help="Directory containing the training data")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
parser.add_argument("--output-dir", type=str, default="model", help="Directory to save the trained model")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Function to load and preprocess images
def load_and_preprocess_images(data_dir):
    images = []
    labels = []
    gesture_mapping = {}
    
    # List all gesture directories
    gesture_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(gesture_dirs)} gesture classes: {gesture_dirs}")
    
    # Create a mapping from gesture names to numeric labels
    for i, gesture in enumerate(sorted(gesture_dirs)):
        gesture_mapping[i] = gesture
    
    # Save the gesture mapping
    with open(os.path.join(args.output_dir, "gesture_mapping.json"), "w") as f:
        json.dump(gesture_mapping, f)
    
    # Load images from each gesture directory
    for label, gesture in gesture_mapping.items():
        gesture_dir = os.path.join(data_dir, gesture)
        
        # Get all jpg files in the directory
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
        print(f"Loading {len(image_files)} images for gesture '{gesture}'")
        
        for img_file in image_files:
            img_path = os.path.join(gesture_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Resize and preprocess image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize to [0, 1]
            
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels), gesture_mapping

# Load and preprocess the data
print("Loading and preprocessing images...")
images, labels, gesture_mapping = load_and_preprocess_images(args.data_dir)

# Convert labels to one-hot encoding
num_classes = len(gesture_mapping)
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_one_hot, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Create data generators with augmentation for training
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Build the model
def build_model(input_shape, num_classes):
    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build and compile the model
model = build_model((224, 224, 3), num_classes)
model.summary()

# Train the model
print(f"Training model for {args.epochs} epochs...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=args.batch_size),
    epochs=args.epochs,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // args.batch_size,
    verbose=1
)

# Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
plt.show()

# Save the Keras model
keras_model_path = os.path.join(args.output_dir, 'model.h5')
model.save(keras_model_path)
print(f"Keras model saved to {keras_model_path}")

# Convert and save the model in TensorFlow.js format
tfjs_model_path = os.path.join(args.output_dir)
tfjs.converters.save_keras_model(model, tfjs_model_path)
print(f"TensorFlow.js model saved to {tfjs_model_path}")

print("Training complete!")