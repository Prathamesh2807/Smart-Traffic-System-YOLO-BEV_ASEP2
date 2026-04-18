import cv2
from ultralytics import YOLO

# Load YOLO model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# Open video file (put your video in same folder)
cap = cv2.VideoCapture("traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Loop through detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # Class names (COCO dataset)
            label = model.names[cls]

            # Filter only vehicles
            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                
    frame = cv2.resize(frame, (900, 600))

    # Show frame
    cv2.imshow("Traffic Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
