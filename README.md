# Drowsiness Detection Model

This project implements a **real-time drowsiness detection system** using a custom-trained YOLOv5 model. The system is deployed as a web application using Streamlit.

## Features
- Detects "awake" and "drowsy" states in real time.
- Uses YOLOv5 for efficient object detection.
- Displays real-time webcam feed with detections rendered on the frames.
- Lightweight and easy-to-use Streamlit interface.

## Requirements
Make sure you have the following installed:

- Python 3.8+
- Torch
- Streamlit
- OpenCV
- NumPy
- Pillow

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
The YOLOv5 model is trained on labeled "awake" and "drowsy" images. To train the model:
1. Collect images for each label (`awake` and `drowsy`).
2. Organize the data and prepare a `dataset.yaml` file.
3. Use YOLOv5's training script to train the model:
   ```bash
   cd yolov5
   python train.py --img 320 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --workers 2
   ```
4. Save the trained weights (`last.pt`) in the specified path.

### 2. Run the Streamlit App
Start the Streamlit app by running:
```bash
streamlit run app.py
```

### 3. Use the App
1. Open the URL provided by Streamlit in your browser.
2. Click the "Start Webcam" checkbox to activate the webcam.
3. View real-time drowsiness detection.

## File Structure
```
.
├── app.py                 # Streamlit application code
├── yolov5/runs/train/exp9/weights/last.pt  # Trained YOLOv5 weights
├── requirements.txt       # Dependencies for the project
└── README.md              # Project documentation
```

## Notes
- Ensure your webcam is connected and accessible.
- To improve performance, the video frames are resized to 640x480.
- Press the 'q' key or uncheck the "Start Webcam" checkbox to stop the webcam feed.

## Future Enhancements
- Add support for additional classes.
- Integrate notifications for detected drowsiness.
- Deploy the app as a cloud service.

---

Developed with 💻 and 🧠.

