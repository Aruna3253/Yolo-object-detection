import cv2
from ultralytics import YOLO

# Load a pre-trained YOLO model

model = YOLO('yolov8s.pt') 

# Open webcam 
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting real-time detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLO Real-Time Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
