# Real-Time Emotion Detection from Facial Landmarks using MATLAB



---

## Project Overview

This project presents a **real-time emotion detection system** developed in **MATLAB** using **facial landmark analysis** and **machine learning**.  
The system captures live video from a webcam, detects facial landmarks (eyes, mouth, nose, eyebrows), and classifies human emotions such as **happy, sad, angry, surprise, fear, disgust, and neutral**.

It demonstrates how MATLAB’s **Computer Vision Toolbox** and **Deep Learning Toolbox** can be leveraged for **affective computing** and **human-computer interaction (HCI)**.

---

## Objectives

- Detect and analyze facial expressions in real time using MATLAB.  
- Extract facial landmarks and compute geometric relationships.  
- Train a classifier (AlexNet/SVM) on emotion datasets (FER-2013, CK+).  
- Achieve accurate emotion classification and real-time feedback.  
- Provide a user-friendly GUI for real-time monitoring.

---

## System Architecture

### Workflow:
1. **Image Acquisition** – Capture real-time frames from webcam.  
2. **Face Detection** – Using Viola–Jones algorithm.  
3. **Preprocessing** – Resize, normalize, and equalize images.  
4. **Facial Landmark Extraction** – Identify 68-point landmarks via Dlib/MATLAB.  
5. **Feature Engineering** – Compute geometric features (distances, angles).  
6. **Emotion Classification** – Classify emotions using AlexNet (transfer learning) or SVM.  
7. **Real-Time Display** – Show detected emotion label on live video.

---

## Datasets Used

### CK+ Dataset
- 500+ sequences from 123 subjects.  
- 7 emotion classes: anger, disgust, fear, happiness, sadness, surprise, neutral.  
- Controlled lighting and facial expressions.

### FER-2013 Dataset
- 35,887 grayscale images (48×48 pixels).  
- Real-world dataset from Kaggle.  
- Includes lighting variation, occlusions, and different poses.  
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/msambare/fer2013/data)

---

## Technologies and Toolboxes

| Toolbox | Purpose |
|----------|----------|
| Image Processing Toolbox | Face detection and preprocessing |
| Computer Vision Toolbox | Webcam and live feed integration |
| Deep Learning Toolbox | Transfer learning with AlexNet |
| Statistics and ML Toolbox | SVM, k-NN, and Random Forest classifiers |
| App Designer | GUI for real-time detection |

---

## Requirements

- MATLAB R2021a or later  
- **Deep Learning Toolbox**  
- **Computer Vision Toolbox**  
- **AlexNet Support Package**  
- **Webcam Support Package**

---

## Project Structure

```
MATLAB Drive/
└── Matlab_project/
    └── emotion_data/
        ├── train/
        │   ├── happy/
        │   ├── sad/
        │   ├── angry/
        │   ├── surprise/
        │   ├── neutral/
        │   ├── fear/
        │   └── disgust/
        └── test/
            ├── happy/
            ├── sad/
            ├── angry/
            ├── surprise/
            ├── neutral/
            ├── fear/
            └── disgust/
```

---

## Training and Evaluation

**Model:** Transfer Learning using **AlexNet**  
**Optimizer:** Stochastic Gradient Descent (SGDM)  
**Learning Rate:** 0.001  
**Epochs:** 10  
**Batch Size:** 64  

The model achieved an average **accuracy of 85%** on test data.

| Emotion | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Happy | 0.94 | 0.93 | 0.93 |
| Sad | 0.90 | 0.89 | 0.89 |
| Angry | 0.91 | 0.90 | 0.90 |
| Surprised | 0.93 | 0.92 | 0.92 |
| Neutral | 0.89 | 0.88 | 0.88 |

---

## Real-Time Emotion Detection

After training, the system:
- Captures live frames via webcam.  
- Detects face regions using **CascadeObjectDetector**.  
- Classifies each detected face in real-time.  
- Displays detected emotion on video feed.  

Example emotions detected:
> 😊 Happy | 😠 Angry | 😢 Sad | 😲 Surprise | 😐 Neutral

---

## Sample MATLAB Commands

**Run the project:**
```matlab
doTraining = false; % Use pre-trained model
run('emotion_detection_script.m');
```

**Train from scratch:**
```matlab
doTraining = true; % Enable training mode
run('emotion_detection_script.m');
```

---

##  Results

- **Overall accuracy:** ~85%  
- **Frame rate:** ~15 FPS  
- **Latency per frame:** <60 ms  
- **Robustness:** Works under normal lighting with moderate head movement.

---

## Applications

- 🎮 **Human–Computer Interaction (HCI)**  
- 🧍 **Healthcare and Therapy Monitoring**  
- 🚗 **Driver Alertness Systems**  
- 🧑‍🏫 **Adaptive E-Learning Systems**  
- 🛡️ **Smart Surveillance and Security**  
- 📈 **Marketing and Consumer Behavior Analysis**

---

## 🚀 Future Enhancements

- Integration of **deep learning (CNN/RNN)** models  
- Multimodal emotion recognition (facial + voice)  
- Mobile and embedded platform deployment (Jetson/Raspberry Pi)  
- Handling occlusions and lighting variations  
- Cultural and demographic generalization  
- Emotion intensity and temporal tracking  

---

## References

1. [Facial Expression Recognition (FER) using ML-HDG – MATLAB Central](https://www.mathworks.com/matlabcentral/fileexchange/166321-facial-expression-recognition-fer-using-ml-hdg)  
2. [EMOTION DETECTION BASED ON FACIAL EXPRESSION – IJCRT, 2022](https://ijcrt.org/papers/IJCRT2209095.pdf)  
3. [Transfer Learning Using AlexNet – MathWorks](https://in.mathworks.com/help/deeplearning/ug/transfer-learning-using-alexnet.html)  
4. [Micro-Facial Expression Recognition – arXiv](https://arxiv.org/abs/2009.13792)  
5. [FER-2013 Dataset – Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)  
6. [Face, Age, and Emotion Detection – MATLAB Central](https://www.mathworks.com/matlabcentral/fileexchange/71819-face-age-and-emotion-detection)

---

