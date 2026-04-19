import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# Define ROI (you will adjust later)
roi_points = np.array([
    [400, 720],  # bottom-left
    [1200, 720],  # bottom-right
    [600, 245],  # top-right
    [450, 245]   # top-left
], np.int32)

# Open video file (put your video in same folder)
cap = cv2.VideoCapture("C:/Users/Prathamesh/Desktop/Smart-Traffic-System-YOLO-BEV_ASEP2/data/traffic_test.mp4")

if not cap.isOpened():
    print("❌ ERROR: Video not found or path wrong")
    exit()
else:
    print("✅ Video loaded successfully")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.resize(frame, (900, 600))


    # Run YOLO detection
    results = model(frame)

    # Draw ROI on frame
    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)

    # Loop through detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # Class names (COCO dataset)
            label = model.names[cls]

            # Filter only vehicles
            if label in ["car", "truck", "bus", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Find center point of detected object
                cx = int((x1 + x2) / 2)
                cy = int(y2)

                # ROI area (you can adjust these values later)
                inside = cv2.pointPolygonTest(roi_points, (cx, cy), False)

                 # Check if object is inside ROI
                if inside >= 0:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0), 2)
                    # Draw label
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 0), 2)
                


    # Show frame
    cv2.imshow("YOLO Traffic Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
