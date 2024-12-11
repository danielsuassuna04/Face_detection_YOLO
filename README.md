# Real-Time Face Detection with YOLO and OpenCV

This repository provides a Python implementation for real-time face detection using a YOLO model and OpenCV. The program uses the computer's webcam to capture video frames, apply YOLO-based predictions, and display the annotated frames in real-time.

## Features
- **Real-Time Face Detection**: Utilizes a YOLO model to detect faces directly from the webcam feed.
- **Interactive Display**: The processed frames with detection boxes and labels are displayed in a live feed.
- **Configurable Confidence Threshold**: Adjust the confidence threshold for predictions as needed.

## Prerequisites

Before running the program, ensure you have the following installed:

- Python 3.8 or higher
- OpenCV
- Ultralytics YOLO library
- A trained YOLO model for face detection (e.g., `face_detection_2.pt`)

Install the required Python packages:

```bash
pip install ultralytics opencv-python
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/danielsuassuna04/Face_detection_YOLO.git
```

2. Place your YOLO model file (e.g., `face_detection_2.pt`) in the same directory as the script.

3. Run the script:

```bash
python face_detection.py
```

4. The webcam feed will open in a new window displaying the real-time face detections.

5. Press `q` to quit the program.

## Code Overview

### Main Components

1. **Model Loading**:
   The YOLO model is loaded using the `Ultralytics` library:
   ```python
   model = YOLO('face_detection_2.pt')
   ```

2. **Webcam Initialization**:
   The program captures the webcam feed using OpenCV:
   ```python
   cap = cv2.VideoCapture(0)
   ```

3. **Real-Time Prediction**:
   Each frame is passed to the YOLO model for prediction:
   ```python
   results = model.predict(frame, conf=0.25)
   ```

4. **Visualization**:
   Detected faces are plotted directly on the frame:
   ```python
   annotated_frame = results[0].plot()
   ```

5. **Interactive Display**:
   The processed frames are displayed in a live feed:
   ```python
   cv2.imshow('YOLO Prediction - Camera', annotated_frame)
   ```

## Customization

- **Confidence Threshold**: Adjust the `conf` parameter in the `model.predict` method to change the confidence threshold for detections:
  ```python
  results = model.predict(frame, conf=0.25)
  ```

- **Model File**: Replace `face_detection_2.pt` with your trained YOLO model file.

## Notes
- Ensure that the YOLO model is trained specifically for face detection to achieve accurate results.
- If the webcam does not initialize, ensure it is connected and not being used by another application.

## License
This project is licensed under the MIT License. Feel free to modify and use it as needed.

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

