import sys
import json

def recognize_sign(image_data):
    # This is where your sign recognition code will go
    # For now, we'll return a dummy gesture
    return "Hello"

if __name__ == "__main__":
    image_data = sys.argv[1]
    recognized_gesture = recognize_sign(image_data)
    print(recognized_gesture)
