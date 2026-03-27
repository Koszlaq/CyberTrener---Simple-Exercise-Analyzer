# 🏋️‍♂️ CyberTrener - Simple Exercise Form Corrector (Multi-Camera)

> A Python application that uses stereoscopic computer vision to analyze and correct exercise form in real-time using two standard webcams.

---

## 📺 Live Demo / Visuals

(SOON)

| Analysis & Feedback |
| :--- |
| ![GIF analysis](path/to/demo_front.gif) |

---

## ✨ Key Features

* **Muli-Camera system:** Syncs and processes live feeds from two webcams for enhanced accuracy.
* **Real-time Form Analysis:** Calculates joint angles (e.g., knee bend during squats) against correct physiological norms.
* **Visual & Audio Feedback:** Instantly alerts the user (on-screen text, potential sound) when incorrect form is detected.
* **Exercise Variety:** Currently supports: **Biceps Curl**, **Barbell Overhead Tricep Extension**, **Barbell Overhead Press**, **Barbell Squat**.
* **Reputation Counting:** Automatically counts correctly executed repetitions.

---

## 🛠️ Tech Stack & Key Concepts

**Languages & Libraries:**
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* **OpenCV:** Image processing, camera synchronization.
* **MediaPipe:** Real-time human pose estimation (keypoint detection).
* **NumPy:** Mathematical operations on coordinates.
* **CustomTkinter:** GUI for our application.

**Computer Vision Concepts Learned/Used:**
1.  **Multi-camera Synchronization:** Handling lags between two USB devices.
2.  **Epipolar Geometry:** Understanding relationship between views.
3.  **Geometric Angle Calculation:** Vector mathematics to determine joint angles in 3D space.
4.  **Front-end:** Creating a user-friendly GUI.
---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.9
* **Two** USB webcams connected to your PC.

### Step-by-step Installation
```bash
# Clone this repository
git clone [link]

# Enter the directory
cd CyberTrener

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
