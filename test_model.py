import os
import numpy as np
import cv2
import torch
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def load_gesture_mapping():
    """Load the gesture mapping from the model directory."""
    with open('model/gesture_mapping.json', 'r') as f:
        return json.load(f)

def load_model():
    """Load the trained PyTorch model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model architecture from train_simple_cnn.py
    from train_simple_cnn import SimpleCNN
    
    # Get number of classes from gesture mapping
    gesture_mapping = load_gesture_mapping()
    num_classes = len(gesture_mapping)
    
    # Create model instance
    model = SimpleCNN(num_classes).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load('model/model.pth', map_location=device))
    
    # Set model to evaluation mode
    model.eval()
    
    return model, device

def preprocess_image(img_path, size=(64, 64)):
    """Preprocess an image for model prediction."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Convert to RGB and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    
    # Convert to PyTorch format (C, H, W) and normalize
    img = img.transpose(2, 0, 1) / 255.0
    
    # Add batch dimension and convert to tensor
    img_tensor = torch.FloatTensor(img).unsqueeze(0)
    
    return img_tensor

def generate_test_image(size=(64, 64)):
    """Generate a test image with hand landmarks for testing."""
    # Create a black image
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)
    
    # Define some sample hand landmarks (normalized coordinates)
    landmarks = [
        (0.5, 0.7),  # Wrist
        (0.45, 0.6), (0.4, 0.5), (0.35, 0.4), (0.3, 0.3),  # Thumb
        (0.5, 0.6), (0.5, 0.5), (0.5, 0.4), (0.5, 0.3),  # Index finger
        (0.55, 0.6), (0.55, 0.5), (0.55, 0.4), (0.55, 0.3),  # Middle finger
        (0.6, 0.6), (0.6, 0.5), (0.6, 0.4), (0.6, 0.3),  # Ring finger
        (0.65, 0.6), (0.65, 0.5), (0.65, 0.4), (0.65, 0.3)   # Pinky finger
    ]
    
    # Draw landmarks as white dots
    for x, y in landmarks:
        draw.ellipse([
            (x * size[0] - 2, y * size[1] - 2), 
            (x * size[0] + 2, y * size[1] + 2)
        ], fill='white')
    
    # Draw connections between landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (0, 5), (5, 9), (9, 13), (13, 17)  # Palm connections
    ]
    
    for i, j in connections:
        draw.line([
            (landmarks[i][0] * size[0], landmarks[i][1] * size[1]),
            (landmarks[j][0] * size[0], landmarks[j][1] * size[1])
        ], fill='white', width=1)
    
    # Save the image
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    img_path = os.path.join(test_dir, 'test_hand.jpg')
    img.save(img_path)
    
    return img_path

def predict(model, img_tensor, device):
    """Make a prediction with the model."""
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        # Get predicted class
        _, predicted = torch.max(outputs, 1)
        # Get confidence scores (softmax)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted[0]].item()
        
    return predicted.item(), confidence

def main():
    try:
        # Load gesture mapping
        gesture_mapping = load_gesture_mapping()
        print("Gesture mapping loaded:", gesture_mapping)
        
        # Load the trained model
        model, device = load_model()
        print("Model loaded successfully")
        
        # Try with a generated test image
        img_path = generate_test_image()
        print(f"Generated test image at {img_path}")
        
        # Get model prediction
        img_tensor = preprocess_image(img_path)
        class_idx, confidence = predict(model, img_tensor, device)
        
        # Map prediction to gesture name
        predicted_gesture = gesture_mapping.get(str(class_idx), "Unknown")
        
        # Display results
        print(f"Predicted gesture: {predicted_gesture}")
        print(f"Confidence: {confidence:.4f}")
        
        # Display the test image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_gesture} ({confidence:.2f})")
        plt.axis('off')
        plt.savefig('test_prediction.png')
        plt.show()
        
        # Try with another generated pose
        alt_img_path = 'test_images/alt_hand.jpg'
        # Create a different hand pose
        img = Image.new('RGB', (64, 64), color='black')
        draw = ImageDraw.Draw(img)
        
        # Different landmarks (mimicking "Hello" gesture with index finger up)
        landmarks = [
            (0.5, 0.7),  # Wrist
            (0.45, 0.65), (0.4, 0.6), (0.35, 0.55), (0.3, 0.5),  # Thumb down
            (0.5, 0.6), (0.5, 0.5), (0.5, 0.4), (0.5, 0.3),  # Index finger up
            (0.55, 0.65), (0.55, 0.7), (0.55, 0.75), (0.55, 0.8),  # Middle finger down
            (0.6, 0.65), (0.6, 0.7), (0.6, 0.75), (0.6, 0.8),  # Ring finger down
            (0.65, 0.65), (0.65, 0.7), (0.65, 0.75), (0.65, 0.8)   # Pinky finger down
        ]
        
        # Draw landmarks and connections
        for x, y in landmarks:
            draw.ellipse([
                (x * 64 - 2, y * 64 - 2), 
                (x * 64 + 2, y * 64 + 2)
            ], fill='white')
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 5), (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        for i, j in connections:
            draw.line([
                (landmarks[i][0] * 64, landmarks[i][1] * 64),
                (landmarks[j][0] * 64, landmarks[j][1] * 64)
            ], fill='white', width=1)
        
        img.save(alt_img_path)
        
        # Get model prediction for the second image
        img_tensor = preprocess_image(alt_img_path)
        class_idx, confidence = predict(model, img_tensor, device)
        
        # Map prediction to gesture name
        predicted_gesture = gesture_mapping.get(str(class_idx), "Unknown")
        
        # Display results
        print(f"Second image - Predicted gesture: {predicted_gesture}")
        print(f"Second image - Confidence: {confidence:.4f}")
        
        # Display the test image
        img = cv2.imread(alt_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_gesture} ({confidence:.2f})")
        plt.axis('off')
        plt.savefig('test_prediction2.png')
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 