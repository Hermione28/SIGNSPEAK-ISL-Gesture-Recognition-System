# SignSpeak – ISL Gesture Recognition System 



## **Description**

This project aims to detect and recognize Indian Sign Language (ISL) gestures using **Mediapipe**, **OpenCV**, and **Machine Learning**. The system extracts hand landmarks in real time and classifies them using a trained neural network. The repository includes code for dataset preprocessing, keypoint extraction, model training, and real-time ISL gesture prediction.

---

## **Overview**

**Dataset:** Indian Sign Language Dataset – Kaggle
**Programming Language:** Python
**Libraries Used:** Mediapipe, OpenCV, NumPy, TensorFlow/Keras

**Dataset Link:**
[https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)

---
**Project Structure**

SignSpeak/
├── dataset/                         # Raw ISL gesture images
├── images/                          # Sample images for reference
├── keypoint.csv                     # Extracted 42 hand landmarks in CSV format

├── ISL_classifier.ipynb             # Model training Jupyter notebook
├── model.h5                         # Trained gesture recognition model

├── isl_detection.py                 # Real-time detection and prediction script
├── dataset_keypoint_generation.py   # Script to generate hand keypoints from dataset

├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
---

## **How It Works**

The system uses **Mediapipe Hands** to detect hand and finger landmarks from webcam input in real time. These extracted **42 keypoints** are fed into a trained **feedforward neural network (FNN)**, which predicts the ISL gesture class.

**Workflow:**

1. Webcam captures a live video frame.
2. Mediapipe detects the hand and extracts 21 keypoints per hand.
3. Extracted coordinates are normalized and passed to the trained classifier.
4. The model predicts the gesture class.
5. The predicted result is displayed on the video stream in real time.

image (gesture process)

---

## **Requirements**

* Python 3.6 or higher
* Mediapipe
* OpenCV
* Numpy
* TensorFlow / Keras

---

## **Installation**

1. Install Python (3.6+).
2. Run the following commands:

```
pip install mediapipe
pip install opencv-python
pip install numpy
pip install tensorflow
```

---

## **Usage**

1. Clone the repository.
2. Open a terminal in the project directory.
3. Run the real-time detection script:

```
python isl_detection.py
```

4. Press **‘q’** to exit the program.

---

## **Examples**

![Example1](ASSETS/Example1.png)
![Example2](ASSETS/Example2.png)

---

## **Next Steps**

⚙️ **Accuracy Improvement:**
Experiment with CNN, LSTM, or hybrid models to improve classification accuracy.

📂 **Dataset Expansion:**
Add more samples and new gesture categories.

🔊 **Speech/Text Output:**
Convert recognized gestures into text or speech for communication support.

🖥️ **GUI Integration:**
Add a user-friendly interface for real-time interaction.

🚀 **Model Deployment:**
Deploy the system as a web application or Android app using TensorFlow Lite.

🤝 **Contributions:**
Fork the repository, create a new branch, and submit a pull request.
Issues can be opened for bugs, enhancements, or new features.

---

## **Acknowledgments**

* Dataset sourced from **Kaggle – Indian Sign Language Dataset**.
* Thanks to the Mediapipe and TensorFlow teams for powerful open-source tools.

---

## **Author**

Prajakta Jagdale

[LinkedIn](www.linkedin.com/in/prajakta-jagdale-665a0a257)

[GitHub](https://github.com/Hermione28)

---

## **About**

A real-time ISL detection system developed using **Mediapipe** and **Machine Learning**.
Includes dataset processing, landmark extraction, model training, and real-time prediction — useful for gesture recognition, accessibility tools, and ISL communication.

---

