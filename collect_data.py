import cv2
import mediapipe as mp
import numpy as np
import os
import time
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Collect hand gesture data for ISL recognition")
parser.add_argument("--gesture", type=str, required=True, help="Name of the gesture to collect data for")
parser.add_argument("--samples", type=int, default=200, help="Number of samples to collect")
parser.add_argument("--output_dir", type=str, default="training_data", help="Directory to save the collected data")
args = parser.parse_args()

# Create output directory if it doesn't exist
gesture_dir = os.path.join(args.output_dir, args.gesture)
os.makedirs(gesture_dir, exist_ok=True)

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up window
cv2.namedWindow("Hand Gesture Collection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Gesture Collection", 800, 600)

# Initialize variables
sample_count = 0
countdown_time = 5  # seconds for countdown
start_time = None
collecting = False
countdown_started = False

print(f"Preparing to collect data for gesture: {args.gesture}")
print("Position your hand in the frame and press 'S' to start collection")

# Progress bar for sample collection
pbar = None

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    
    # Display countdown
    if countdown_started and not collecting:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, countdown_time - elapsed_time)
        
        if remaining_time > 0:
            cv2.putText(
                frame, 
                f"Starting in: {int(remaining_time) + 1}", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
        else:
            collecting = True
            pbar = tqdm(total=args.samples, desc=f"Collecting {args.gesture} samples")
            print(f"Starting collection for {args.gesture}. Keep your hand steady.")
    
    # Show instructions when not collecting
    if not countdown_started and not collecting:
        cv2.putText(
            frame,
            "Press 'S' to start collection",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )
    
    # If collecting and hand is detected, save the sample
    if collecting and results.multi_hand_landmarks:
        # Get hand landmarks
        landmarks = results.multi_hand_landmarks[0]
        
        # Save frame as image
        img_path = os.path.join(gesture_dir, f"{args.gesture}_{sample_count:04d}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Extract and save landmarks as numpy array
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        landmarks_path = os.path.join(gesture_dir, f"{args.gesture}_{sample_count:04d}.npy")
        np.save(landmarks_path, landmarks_array)
        
        sample_count += 1
        if pbar:
            pbar.update(1)
        
        # Stop collection when enough samples are collected
        if sample_count >= args.samples:
            if pbar:
                pbar.close()
            print(f"Collected {sample_count} samples for {args.gesture}")
            break
        
        # Small delay to avoid duplicate frames
        time.sleep(0.1)
    
    # Show sample count if collecting
    if collecting:
        cv2.putText(
            frame,
            f"Samples: {sample_count}/{args.samples}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    # Display the frame
    cv2.imshow("Hand Gesture Collection", frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Start countdown when 'S' is pressed
    if key == ord('s') and not countdown_started and not collecting:
        countdown_started = True
        start_time = time.time()
        print(f"Starting countdown for {args.gesture} data collection...")
    
    # Exit on 'Q' press
    if key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"Data collection complete. {sample_count} samples saved to {gesture_dir}")
