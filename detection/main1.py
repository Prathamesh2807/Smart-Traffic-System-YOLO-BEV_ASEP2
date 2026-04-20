import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort

# Load YOLO model (downloads automatically first time)
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.2)

line_y = 425   # adjust based on your road position
count = 0
counted_ids = set()   # to avoid duplicate counting (basic fix)
previous_positions = {}

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

    # 🔥 STEP 3: create empty detections list
    detections = []

    # 🔥 DRAW COUNTING LINE (STEP 2)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)


    # Run YOLO detection
    results = model(frame)

    # Draw ROI on frame
    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)



    # YOLO LOOP
    # Loop through detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # Class names (COCO dataset)
            label = model.names[cls]

            # Filter only vehicles
            if label in ["car", "truck", "bus", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # store detection for SORT
                detections.append([x1, y1, x2, y2, conf])

            # 🔥 STEP 5: SEND TO SORT
            if len(detections) > 0:
                detections_np = np.array(detections)
            else:
                detections_np = np.empty((0, 5))

            tracked_objects = tracker.update(detections_np)


            # 🔥 STEP 6: DRAW + COUNT USING SORT
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = map(int, obj)

                cx = int((x1 + x2) / 2)
                cy = int(y2)

                inside = cv2.pointPolygonTest(roi_points, (cx, cy), False)

                if inside >= 0:
                    # draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # show ID instead of label
                    cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0,255,0), 2)

                    # center point
                    cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

                    # 🔥 NEW COUNT LOGIC
                    prev_cy = previous_positions.get(obj_id, None)

                    if prev_cy is not None:
                        if prev_cy < line_y and cy >= line_y :
                            if obj_id not in counted_ids:
                                count += 1
                                counted_ids.add(obj_id)

                                cv2.putText(frame, "COUNTED", (x1, y1 - 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                           (0, 0, 255), 2)

                    previous_positions[obj_id] = cy
                    

                
    cv2.putText(frame, f"Count: {count}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO Traffic Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
