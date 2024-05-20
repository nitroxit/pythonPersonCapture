import cv2
from ultralytics import YOLO
import os
import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Create images directory if it doesn't exist
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform detection
    results = model(frame)
    
    for result in results:
        boxes = result.boxes.xyxy  # bounding boxes
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if int(class_id) == 0:  # class_id 0 corresponds to 'person'
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Label the box with confidence
                label = f'Person: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Save the image with a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = os.path.join(images_dir, f'detected_person_{timestamp}.jpg')
                cv2.imwrite(filename, frame)

    # Display the frame
    cv2.imshow('YOLOv8 Person Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
