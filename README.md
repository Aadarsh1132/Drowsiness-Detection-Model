# Real-Time Drowsiness Detection using YOLOv5 😴 awake

[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Library](https://img.shields.io/badge/Library-OpenCV-informational.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) // Choose an appropriate license

## Overview

This project implements a real-time drowsiness detection system using the powerful **YOLOv5 object detection model** provided by Ultralytics, built on the **PyTorch** framework. The system leverages a webcam feed to monitor a person's state (drowsy vs. awake) by fine-tuning a YOLOv5 model on custom data.

This repository demonstrates key skills in:

* **Computer Vision:** Real-time video processing, object detection.
* **Deep Learning:** Utilizing and fine-tuning state-of-the-art models (YOLOv5).
* **Framework Proficiency:** Implementing solutions using PyTorch.
* **Data Handling:** (Implied) Collecting, labeling (using tools like LabelImg), and training with custom datasets.
* **Practical Application:** Building a system with potential real-world impact (e.g., driver safety).

**(Optional: Add a GIF/Screenshot here showing the detection in action!)**
![Demo GIF Placeholder](link_to_your_demo_gif_or_screenshot.gif)

---

## Features

* **Baseline Object Detection:** Utilizes the pre-trained YOLOv5s model (trained on COCO dataset - 80 classes) for general object detection on images, videos, and webcam feeds.
* **Custom Drowsiness Detection:** Fine-tuned YOLOv5 model specifically trained to detect 'Drowsy' and 'Awake' states in real-time.
* **Real-Time Processing:** Leverages OpenCV to capture and process webcam feeds efficiently.
* **Flexible Input:** Capable of performing detections on:
    * Static Images (from URLs or local files)
    * Video Files
    * Live Webcam Feeds
* **PyTorch Integration:** Built entirely using PyTorch and the Ultralytics YOLOv5 implementation.
* **Visualization:** Renders bounding boxes and class labels directly onto the output feed using OpenCV and Matplotlib.

---

## How It Works

1.  **Setup:** Installs necessary dependencies including PyTorch and clones the Ultralytics YOLOv5 repository.
2.  **Load Model:** Loads either the pre-trained YOLOv5 model from PyTorch Hub or the custom-trained drowsiness detection model.
3.  **Input:** Captures frames from a webcam, video file, or loads a static image.
4.  **Detection:** Passes the input frame/image to the loaded YOLOv5 model for inference.
5.  **Rendering:** Uses the model's output and OpenCV/Matplotlib to draw bounding boxes and labels on the frame.
6.  **Output:** Displays the processed frame in real-time or saves the output.
7.  **Data Collection:** Gather images representing 'Drowsy' and 'Awake' states.
8.  **Labeling:** Annotate images using a tool like LabelImg to create bounding boxes for the classes.
9.  **Training:** Fine-tune the YOLOv5 model on the custom labeled dataset.

---

## Technologies Used

* **Python 3.x**
* **PyTorch:** Deep learning framework.
* **Ultralytics YOLOv5:** The specific YOLOv5 implementation used.
* **OpenCV (cv2):** Library for real-time computer vision tasks (image/video reading, processing, display).
* **Matplotlib:** Library for plotting and rendering images within the notebook.
* **NumPy:** Library for numerical operations and array manipulation.
* **Jupyter Notebook:** For development and demonstration.
* **(Mention LabelImg if used for custom data labeling)**

---

## Setup and Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install PyTorch:**
    * Visit the official [PyTorch website](https://pytorch.org/get-started/locally/).
    * Select your preferences (OS, Package Manager, Compute Platform - CUDA recommended for faster training/inference).
    * Run the generated installation command. Example (check website for current):
        ```bash
        # Example for pip, Linux/Windows, CUDA 11.x - VERIFY ON PYTORCH.ORG
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
        ```

3.  **Clone Ultralytics YOLOv5 & Install Requirements:**
    * The transcript clones this separately. You might include it as a submodule or directly in your repo. Assuming it's cloned separately as per the transcript:
        ```bash
        git clone [https://github.com/ultralytics/yolov5.git](https://github.com/ultralytics/yolov5.git)
        cd yolov5
        pip install -r requirements.txt
        cd ..
        ```
    * *(Alternatively, if you include `requirements.txt` in your main repo)*:
        ```bash
        pip install -r requirements.txt
        ```
        *(Ensure this requirements file includes all necessary packages like opencv-python, matplotlib, numpy, etc., in addition to those needed by YOLOv5 if you integrate it differently).*

---

## Usage

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open the Notebook:** Navigate to and open the main `.ipynb` file (e.g., `Drowsiness_Detection.ipynb`).

3.  **Run the Cells:** Execute the cells sequentially.
    * **Installation & Imports:** The initial cells handle setup and library imports.
    * **Load Model:** Load the pre-trained YOLOv5s model (`torch.hub.load('ultralytics/yolov5', 'yolov5s')`).
    * **Image Detection:** Test baseline detection on sample image URLs or local paths.
    * **Video/Real-Time Baseline Detection:** Run the OpenCV loop to test detection on a video file or live webcam feed using the *baseline* COCO model.
    * **(Optional) Train Custom Model:** Execute cells related to loading custom data, setting up configuration, and training the drowsiness detector (if included in the notebook).
    * **Load Custom Model:** Load your fine-tuned drowsiness detection weights (e.g., `path/to/your/best.pt`).
    * **Real-Time Drowsiness Detection:** Run the OpenCV loop using the *custom* drowsiness model for live detection via webcam. Press 'q' to quit the feed window.

---

## Potential Applications

* **Driver Safety:** Alerting drivers when signs of drowsiness are detected.
* **Operator Monitoring:** Ensuring alertness in operators of heavy machinery or critical systems.
* **Security:** Monitoring attention levels in surveillance contexts.
* **Research:** Studying attention and fatigue patterns.

---

## Acknowledgements

* This project was inspired by and follows the structure outlined in the tutorial by Nicholas Renaut.
* Utilizes the YOLOv5 model implementation by [Ultralytics](https://github.com/ultralytics/yolov5).

---

**(Optional: Add Contact Info / Contribution Guidelines)**

Feel free to reach out with questions or suggestions!
