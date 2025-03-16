import tensorflow as tf
import numpy as np
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_landmarks(image_path):
    """Extract hand landmarks from an image using MediaPipe."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(image)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get landmarks of the first hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Convert landmarks to flat array
    landmarks_array = []
    for landmark in hand_landmarks.landmark:
        landmarks_array.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks_array)

def load_dataset(data_dir='training_data'):
    """Load and preprocess the dataset."""
    X = []  # Hand landmarks
    y = []  # Labels
    gesture_mapping = {}  # Map gesture names to indices
    
    print("Loading dataset...")
    gestures = sorted(os.listdir(data_dir))  # Sort to ensure consistent order
    
    for idx, gesture in enumerate(gestures):
        gesture_mapping[idx] = gesture
        gesture_dir = os.path.join(data_dir, gesture)
        print(f"Processing {gesture}...")
        
        for image_file in os.listdir(gesture_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(gesture_dir, image_file)
            landmarks = extract_hand_landmarks(image_path)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(idx)
    
    return np.array(X), np.array(y), gesture_mapping

def create_model(num_classes, input_shape):
    """Create the model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history):
    """Plot training history."""
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
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and preprocess data
    X, y, gesture_mapping = load_dataset()
    
    if len(X) == 0:
        print("No valid training data found!")
        return
    
    print(f"\nDataset summary:")
    print(f"Total samples: {len(X)}")
    for idx, gesture in gesture_mapping.items():
        count = np.sum(y == idx)
        print(f"{gesture}: {count} samples")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    num_classes = len(gesture_mapping)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Create and compile model
    model = create_model(num_classes, input_shape=(63,))  # 21 landmarks × 3 coordinates
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model in TensorFlow.js format
    print("\nSaving the model...")
    model_save_path = 'model'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Save gesture mapping
    import json
    with open(os.path.join(model_save_path, 'gesture_mapping.json'), 'w') as f:
        json.dump(gesture_mapping, f)
    
    # Convert and save model
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, model_save_path)
    print(f"Model and gesture mapping saved in '{model_save_path}' directory")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}") 