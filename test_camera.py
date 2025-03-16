import cv2

def test_cameras():
    print("Testing camera access...")
    
    # Try different camera indices
    for i in range(4):
        print(f"\nTrying camera index {i}:")
        
        # Try DirectShow first
        print(f"Testing with DirectShow...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Successfully accessed camera {i} with DirectShow")
                print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                cap.release()
                continue
            else:
                print("✗ Could not read frame with DirectShow")
        else:
            print("✗ Could not open camera with DirectShow")
        cap.release()
        
        # Try default API
        print(f"Testing with default API...")
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Successfully accessed camera {i} with default API")
                print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            else:
                print("✗ Could not read frame with default API")
        else:
            print("✗ Could not open camera with default API")
        cap.release()

if __name__ == "__main__":
    test_cameras() 