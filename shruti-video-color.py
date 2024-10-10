import cv2
import os
import torch
import numpy as np

# ------------------------------------------------------ DEFINING GLOBAL VARIABLES -----------------------------------------------------------------------------
# Path to the custom-trained YOLOv5 model for helmet detection
helmet_weights_path = './bestHelmet.pt'

# Path to the input video for helmet detection
input_video_path = './test_videos/input_video.mp4'

# Path to save the output video with detection annotations
output_video_path = './test_videos/output_video_helmet_detection.mp4'

# Directory to save cropped helmet images
output_cropped_dir = './yolov5-master/cropped_helmets'
os.makedirs(output_cropped_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Detection parameters
helmet_conf_threshold = 0.2  # Confidence threshold for helmet detection
img_size = 416  # Image size for YOLOv5 model input

# ------------------------------------------------------ LOAD YOLOv5 HELMET DETECTION MODEL ---------------------------------------------------------------------
# Load the custom YOLOv5 model using PyTorch's torch.hub. The model is trained to detect helmets.
print("Loading helmet detection model...")
helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path=helmet_weights_path, force_reload=True)

# Set the confidence threshold for the helmet model to filter low-confidence detections
helmet_model.conf = helmet_conf_threshold  
print("Model loaded successfully.")

# ------------------------------------------------------ VIDEO INITIALIZATION ---------------------------------------------------------------------------------
# Initialize video capture with the input video
cap = cv2.VideoCapture(input_video_path)

# Check if the video file can be opened
if not cap.isOpened():
    print(f"Error opening video file {input_video_path}")
    exit()

# Get video properties such as width, height, and frames per second (FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Video properties: Width={frame_width}, Height={frame_height}, FPS={fps}, Total Frames={total_frames}")
print("Starting video processing...")

# Initialize frame count to track the progress of video processing
frame_count = 0

# ------------------------------------------------------ MAIN VIDEO PROCESSING LOOP ---------------------------------------------------------------------------
while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    
    # Break the loop if no frame is available (end of video)
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break

    # Increment frame count for progress tracking
    frame_count += 1
    print(f"Processing frame {frame_count} of {total_frames}")

    # Convert the frame from BGR (OpenCV format) to RGB (YOLOv5 format)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------ HELMET DETECTION ----------------------------------------------------------------------------------
    # Perform helmet detection using the YOLOv5 model (without gradient computation for faster inference)
    with torch.no_grad():
        helmet_results = helmet_model(img)

    # Extract detection results in the format [x1, y1, x2, y2, confidence, class]
    helmets = helmet_results.xyxy[0]

    # Iterate through all detected objects in the frame
    for idx, helmet in enumerate(helmets):
        # Extract bounding box coordinates, confidence score, and class ID
        x_min, y_min, x_max, y_max, conf, cls = helmet
        x_min, y_min, x_max, y_max = map(int, [x_min.item(), y_min.item(), x_max.item(), y_max.item()])
        conf = conf.item()
        cls = int(cls.item())

        print(f"Detection: Class ID {cls}, Confidence {conf:.2f}, BBox: ({x_min}, {y_min}), ({x_max}, {y_max})")

        # Set bounding box color: green for helmets, red for no helmets
        color = (0, 255, 0) if cls == 1 else (0, 0, 255)  # cls == 1 assumed to represent "helmet"
        box_thickness = 3

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, box_thickness)

        # Add a label for the bounding box (helmet or no helmet, with confidence score)
        font_scale = 1.0  # Font size for label
        font_thickness = 2  # Thickness of label text
        label = f'Helmet {idx}: {conf:.2f}' if cls == 1 else f'No Helmet {idx}: {conf:.2f}'
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # ------------------------------------------------------ SAVE CROPPED HELMET IMAGE ---------------------------------------------------------------------
        # Crop the helmet region from the frame
        helmet_crop = img[y_min:y_max, x_min:x_max]
        
        # Save the cropped image if it's not empty
        if helmet_crop.size != 0:
            cropped_helmet_path = os.path.join(output_cropped_dir, f'frame{frame_count}_helmet{idx}.jpg')
            cv2.imwrite(cropped_helmet_path, cv2.cvtColor(helmet_crop, cv2.COLOR_RGB2BGR))
        else:
            print(f"Empty crop for helmet {idx} in frame {frame_count}, skipping saving.")

    # ------------------------------------------------------ WRITE FRAME TO OUTPUT VIDEO ----------------------------------------------------------------------
    # Write the processed frame (with bounding boxes and labels) to the output video
    out.write(frame)

    # ------------------------------------------------------ DISPLAY FRAME LIVE -------------------------------------------------------------------------------
    # Display the frame in a window (for live preview)
    cv2.imshow('Live Helmet Detection', frame)

    # Allow early exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early termination requested. Exiting...")
        break

# ------------------------------------------------------ RELEASE RESOURCES ------------------------------------------------------------------------------------
# Release the video capture and video writer resources
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video processing complete. Output saved to:", output_video_path)
