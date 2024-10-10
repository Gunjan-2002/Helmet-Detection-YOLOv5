import cv2
import os
import torch
import numpy as np

# ------------------------------------------------------ DEFINING GLOBAL VARIABLES --------------------------------------------------------
helmet_weights_path = './bestHelmet.pt'               # Path to helmet detection model
output_cropped_dir = './yolov5-master/cropped_helmets'  # Directory to save cropped helmet images
os.makedirs(output_cropped_dir, exist_ok=True)

# Detection parameters
helmet_conf_threshold = 0.2
img_size = 416  # Image size for YOLOv5 models

# ------------------------------------------------------ LOAD YOLOv5 HELMET DETECTION MODEL ------------------------------------------------
# Load helmet detection model
print("Loading helmet detection model...")
helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path=helmet_weights_path, force_reload=True)
helmet_model.conf = helmet_conf_threshold  # Set confidence threshold
print("Model loaded successfully.")

# ------------------------------------------------------ LIVE CAMERA INITIALIZATION -------------------------------------------------------
# Initialize video capture for live camera (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening the camera")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Live camera properties: Width={frame_width}, Height={frame_height}, FPS={fps}")
print("Starting live helmet detection...")

frame_count = 0

# ------------------------------------------------------ MAIN VIDEO PROCESSING LOOP ------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame from live camera")
        break  # Exit the loop if there is an issue with capturing frames

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Convert frame to RGB (YOLOv5 expects RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------ HELMET DETECTION -----------------------------------------------------
    with torch.no_grad():
        helmet_results = helmet_model(img)

    # Parse helmet detection results
    helmets = helmet_results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

    for idx, helmet in enumerate(helmets):
        x_min, y_min, x_max, y_max, conf, cls = helmet
        x_min = int(x_min.item())
        y_min = int(y_min.item())
        x_max = int(x_max.item())
        y_max = int(y_max.item())
        conf = conf.item()
        cls = int(cls.item())

        print(f"Detection: Class ID {cls}, Confidence {conf:.2f}, BBox: ({x_min}, {y_min}), ({x_max}, {y_max})")

        # Set bounding box color based on the detection class (helmet vs no helmet)
        if cls == 0:  # Assuming 1 is for helmet
            color = (0, 255, 0)  # Green for helmet
        else:
            color = (0, 0, 255)  # Red for no helmet

        box_thickness = 3

        # Draw bounding box and label
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, box_thickness)

        font_scale = 1.0  # Increased font size
        font_thickness = 2  # Increased thickness of the label text

        label = f'Helmet {idx}: {conf:.2f}' if cls == 0 else f'No Helmet {idx}: {conf:.2f}'
        cv2.putText(frame, label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # ------------------------------------------------------ SAVE CROPPED HELMET IMAGE ------------------------------------------------
        helmet_crop = img[y_min:y_max, x_min:x_max]
        if helmet_crop.size != 0:
            cropped_helmet_path = os.path.join(output_cropped_dir, f'frame{frame_count}_helmet{idx}.jpg')
            cv2.imwrite(cropped_helmet_path, cv2.cvtColor(helmet_crop, cv2.COLOR_RGB2BGR))
        else:
            print(f"Empty crop for helmet {idx} in frame {frame_count}, skipping saving.")

    # ------------------------------------------------------ DISPLAY FRAME LIVE -----------------------------------------------------------
    cv2.imshow('Live Helmet Detection', frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early termination requested. Exiting...")
        break

# ------------------------------------------------------ RELEASE RESOURCES ---------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("Live helmet detection stopped.")
