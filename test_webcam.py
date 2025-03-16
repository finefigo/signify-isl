import cv2
import time

def test_webcam():
    print("Testing webcam access...")
    
    # Try different APIs
    apis = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    
    for api in apis:
        print(f"\nTrying API: {api}")
        cap = cv2.VideoCapture(0, api)
        
        if not cap.isOpened():
            print("Failed to open camera with this API")
            continue
            
        print("Camera opened successfully")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        # Try to read frames
        print("Attempting to read frames...")
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}")
                continue
            
            if frame is None:
                print(f"Frame {i} is None")
                continue
                
            # Check if frame is black
            if cv2.mean(frame)[0] < 1.0:
                print(f"Frame {i} is black")
            else:
                print(f"Frame {i} is valid")
                
            # Try to display frame
            try:
                cv2.imshow('Test Frame', frame)
                cv2.waitKey(100)
            except Exception as e:
                print(f"Error displaying frame: {str(e)}")
        
        print("\nClosing camera...")
        cap.release()
        cv2.destroyAllWindows()
        
        response = input("\nDid you see the camera feed? (y/n): ")
        if response.lower() == 'y':
            print(f"Success! Use API {api} for the data collection script.")
            return
    
    print("\nNo working configuration found.")
    print("Please check your webcam connections and permissions.")

if __name__ == "__main__":
    test_webcam() 