import cv2
import mediapipe as mp
import os
import time
from datetime import datetime

def find_working_camera():
    """Initialize camera with DirectShow API which we know works."""
    print("Initializing camera with DirectShow...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            # Try to read a frame to make sure it's working
            for _ in range(5):  # Try a few times as first frame might be black
                ret, frame = cap.read()
                if ret and frame is not None and not is_black_frame(frame):
                    print("Successfully initialized camera with DirectShow")
                    return cap
                time.sleep(0.1)
            cap.release()
            
        print("Failed to initialize with DirectShow, trying default API...")
        cap = cv2.VideoCapture(0)  # Fallback to default API
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and not is_black_frame(frame):
                print("Successfully initialized camera with default API")
                return cap
            cap.release()
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
    
    return None

def is_black_frame(frame):
    """Check if a frame is completely or mostly black."""
    if frame is None:
        return True
    return cv2.mean(frame)[0] < 1.0  # Check average pixel value

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define the gestures
GESTURES = ['Hello', 'Thank You', 'Yes', 'No', 'Please',
            'Sorry', 'Good', 'Bad', 'Name', 'Help']

def create_directories():
    """Create directories for each gesture if they don't exist."""
    for gesture in GESTURES:
        os.makedirs(os.path.join('training_data', gesture), exist_ok=True)

def collect_data():
    print("\nInitializing camera...")
    print("If you see a black screen, please:")
    print("1. Make sure your webcam isn't being used by another application")
    print("2. Try closing and reopening this program")
    print("3. Try unplugging and replugging your webcam if it's external\n")
    
    # Try to find a working camera
    cap = find_working_camera()
    
    if cap is None:
        print("\nError: Could not open any camera!")
        print("Please ensure that:")
        print("1. Your webcam is properly connected")
        print("2. No other application is using the webcam")
        print("3. You have given camera permissions to the application")
        return
    
    try:
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # Set middle brightness
        cap.set(cv2.CAP_PROP_CONTRAST, 128)  # Set middle contrast
        
        current_gesture_index = 0
        image_count = 0
        collecting = False
        timer_start = 0
        black_frame_count = 0
        
        print("\nData Collection Instructions:")
        print("1. Press 'SPACE' to start/stop collecting images for current gesture")
        print("2. Press 'n' to move to next gesture")
        print("3. Press 'p' to move to previous gesture")
        print("4. Press 'q' to quit")
        print("5. Press 'r' to reset camera if screen is black")
        print("\nRecommended: Collect at least 100 images per gesture")
        print("Vary your hand position, angle, and distance from camera\n")
        
        window_name = 'Collect Training Data (Press Q to quit)'
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Could not read frame from camera!")
                black_frame_count += 1
                if black_frame_count > 30:  # If black for 30 frames, try to reset
                    print("Attempting to reset camera...")
                    cap.release()
                    cap = find_working_camera()
                    if cap is None:
                        print("Failed to reset camera!")
                        break
                    black_frame_count = 0
                continue
            
            if is_black_frame(frame):
                black_frame_count += 1
                if black_frame_count > 30:
                    print("Camera showing black frames. Press 'r' to reset or 'q' to quit.")
                continue
            else:
                black_frame_count = 0
            
            try:
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and detect hands
                results = hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                
                # Draw collection area guide
                cv2.rectangle(frame, (100, 100), (540, 380), (0, 255, 0), 2)
                
                # Display current gesture and status
                current_gesture = GESTURES[current_gesture_index]
                status_text = f"Current Gesture: {current_gesture}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                count_text = f"Images: {len(os.listdir(os.path.join('training_data', current_gesture)))}"
                cv2.putText(frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if collecting:
                    # Add recording indicator
                    cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)
                    
                    # Collect image every 100ms if hand is detected
                    if time.time() - timer_start >= 0.1:  # 100ms
                        if results.multi_hand_landmarks:
                            # Save the frame
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = f"{current_gesture}_{timestamp}.jpg"
                            filepath = os.path.join('training_data', current_gesture, filename)
                            cv2.imwrite(filepath, frame)
                            image_count += 1
                            timer_start = time.time()
                
                # Display the frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key == -1:  # No key pressed
                    continue
                    
                key = key & 0xFF  # Convert to ASCII
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord('r'):  # Reset camera
                    print("Resetting camera...")
                    cap.release()
                    cap = find_working_camera()
                    if cap is None:
                        print("Failed to reset camera!")
                        break
                elif key == ord(' '):  # Start/Stop collecting
                    collecting = not collecting
                    timer_start = time.time()
                    if collecting:
                        print(f"Started collecting images for '{current_gesture}'")
                    else:
                        print(f"Stopped collecting images for '{current_gesture}'")
                elif key == ord('n'):  # Next gesture
                    collecting = False
                    current_gesture_index = (current_gesture_index + 1) % len(GESTURES)
                    print(f"\nSwitched to gesture: {GESTURES[current_gesture_index]}")
                elif key == ord('p'):  # Previous gesture
                    collecting = False
                    current_gesture_index = (current_gesture_index - 1) % len(GESTURES)
                    print(f"\nSwitched to gesture: {GESTURES[current_gesture_index]}")
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Release resources
        print("\nCleaning up...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        for i in range(5):  # Sometimes needed to fully close windows on Windows
            cv2.waitKey(1)
        print("Done!")

if __name__ == '__main__':
    try:
        create_directories()
        collect_data()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        cv2.destroyAllWindows()
        for i in range(5):  # Ensure windows are closed
            cv2.waitKey(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        cv2.destroyAllWindows()
        for i in range(5):  # Ensure windows are closed
            cv2.waitKey(1) 