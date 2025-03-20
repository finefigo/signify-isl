import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import tensorflowjs as tfjs

def load_data(data_dir='training_data'):
    """Load images and labels from the training data directory."""
    images = []
    labels = []
    gesture_mapping = {}
    
    # Get all gesture directories
    gesture_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(gesture_dirs)} gesture classes: {gesture_dirs}")
    
    # Create gesture mapping
    for i, gesture in enumerate(gesture_dirs):
        gesture_mapping[str(i)] = gesture
    
    # Load images from each gesture directory
    for label, gesture in enumerate(gesture_dirs):
        gesture_dir = os.path.join(data_dir, gesture)
        print(f"Loading images for gesture '{gesture}'...")
        
        # Get all jpg files
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(gesture_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # Convert to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))  # Resize to smaller size for faster training
                images.append(img)
                labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    return X, y, gesture_mapping

def create_model(input_shape, num_classes):
    """Create a 2D CNN model."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model/training_history.png')
    plt.close()

def main():
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    X, y, gesture_mapping = load_data()
    
    # Save gesture mapping
    with open('model/gesture_mapping.json', 'w') as f:
        json.dump(gesture_mapping, f, indent=4)
    print("Saved gesture mapping")
    
    # Convert labels to one-hot encoding
    num_classes = len(gesture_mapping)
    y = to_categorical(y, num_classes)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Create and compile model
    print("Creating model...")
    model = create_model(input_shape=(64, 64, 3), num_classes=num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Save model in TensorFlow.js format
    print("\nSaving model...")
    tfjs.converters.save_keras_model(model, 'model')
    print("Model saved in TensorFlow.js format")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 