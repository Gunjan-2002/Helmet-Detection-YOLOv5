
#                                                    Helmet Detection Video Processing Project


### Author: Richest man on earth

### Project Overview
This project detects helmets in a video stream using a pre-trained YOLOv5 model. The project processes each frame of the video, detects objects classified as helmets or non-helmets, draws bounding boxes around them, saves cropped helmet images, and generates an output video with the detections displayed.

---

### Table of Contents
1. [Project Description]
2. [How It Works]
3. [Installation & Prerequisites]
4. [Code Overview]
5. [Running the Project]

---

### Project Description

This project leverages a YOLOv5-based deep learning model to detect helmets in a given video. The key components of the project include:
- Loading a custom YOLOv5 model trained specifically to detect helmets.
- Processing each frame of an input video to detect helmets and annotate frames.
- Saving cropped images of detected helmets to a specified directory.
- Outputting an annotated video highlighting the detections.
  
---

### How It Works

1. **Model Loading**: A custom-trained YOLOv5 model (`bestHelmet.pt`) is loaded to detect helmets in video frames. The model is configured with a confidence threshold for detections.
   
2. **Video Processing**: The input video is processed frame by frame. Each frame is passed through the YOLOv5 model for helmet detection.

3. **Detection & Visualization**: For each detection, bounding boxes are drawn around helmets (green) or non-helmets (red), with a confidence score displayed.

4. **Cropped Image Saving**: Cropped images of detected helmets are saved to a specified directory.

5. **Output Video Generation**: The frames with drawn bounding boxes and labels are written into a new video file.

---

### Installation & Prerequisites

To run this project, you need the following prerequisites installed:

#### 1. Python 3.x
Make sure Python 3.x is installed on your machine. You can download it [here](https://www.python.org/downloads/).

#### 2. Required Libraries
Install the required Python libraries using pip:
```bash
pip install torch torchvision opencv-python opencv-python-headless
```

**Required Libraries Overview**:
- **OpenCV (`cv2`)**: Used for video processing, drawing bounding boxes, and saving cropped images.
- **PyTorch (`torch`)**: Used for loading and running the YOLOv5 model.
- **NumPy (`numpy`)**: Used for image manipulation.

#### 3. YOLOv5 Model
You need a custom-trained YOLOv5 model (`bestHelmet.pt`) to run the helmet detection. This model should be pre-trained to recognize helmets.

#### 4. Video Input
An input video (`input_video.mp4`) must be placed in the `./test_videos/` directory.

---

### Code Overview

The project consists of the following key sections:

1. **Global Variables**:
   - Define paths for the model, input video, and output video.
   - Set detection parameters such as the confidence threshold and image size.

2. **Model Loading**:
   - Load the YOLOv5 helmet detection model using PyTorch's `torch.hub.load`.
   - Set the model's confidence threshold.

3. **Video Initialization**:
   - Use OpenCV to open and read the input video.
   - Get properties like frame width, height, frames per second (FPS), and total frame count.
   - Create a `VideoWriter` object to save the processed video.

4. **Main Processing Loop**:
   - Loop through each frame of the video.
   - Convert each frame to RGB format (required for YOLOv5).
   - Perform helmet detection using the YOLOv5 model.
   - Draw bounding boxes and labels on detected helmets.
   - Save cropped helmet images.
   - Write the processed frame to the output video.

5. **Saving and Cleanup**:
   - Save the output video and release all resources used by OpenCV.
   - Provide an option to quit the video processing early by pressing the 'q' key.

---

### Running the Project

To run the project, follow these steps:

1. **Prepare the Environment**: Ensure all required libraries are installed and the YOLOv5 model is available in the specified path (`./bestHelmet.pt`).

2. **Place Input Video**: Put the input video file (`input_video.mp4`) in the `./test_videos/` directory.

3. **Run the Code**:
   - Run the Python script with the following command:
   ```bash
   python helmet_detection.py
   ```

4. **View the Output**: After processing, the output video with helmet detection will be saved in the `./test_videos/` directory as `output_video_helmet_detection.mp4`.

---

### Note:
- The detection model (`bestHelmet.pt`) is trained for specific detection scenarios. Ensure that the model is suitable for your use case.
- The video processing might be resource-intensive depending on the video size and frame rate.

---

This documentation should give users a clear understanding of the project and how to set it up and run it.