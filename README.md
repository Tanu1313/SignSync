
# SignSync: AI-Powered Sign Language Detector

**SignSync** is a real-time sign language detection system that translates hand gestures into text, bridging communication gaps for individuals with hearing or speech impairments. The project leverages **computer vision**, **machine learning**, and an **interactive web interface** for instant feedback and usability.

---

## 🚀 Features

### Real-Time Gesture Recognition
- Tracks hand gestures using **MediaPipe’s 21 hand landmarks** in live video.  
- Processes frames in real-time for accurate gesture recognition.

### Webcam-Based Input
- Captures input using a standard webcam; no additional hardware required.  
- Continuously monitors hand positions and shapes for detection.

### Machine Learning-Based Classification
- Uses **RandomForestClassifier** to classify hand gestures from landmarks.  
- No deep learning (CNN) required—lightweight and fast.

### Text-Only Output
- Converts recognized gestures into on-screen text.  
- Provides clear communication without relying on audio.

### Streamlit Web Interface
- Interactive frontend to view live webcam feed.  
- Displays detected gestures in real-time with hand landmarks visualization.

### Live Feedback via OpenCV
- Draws bounding boxes and landmarks on the video feed.  
- Useful for user guidance and debugging.

### Trainable and Extendable
- Easily add new gestures by collecting more data.  
- Model can be retrained and saved using **Pickle**.

### Accessibility-Focused
- Helps individuals with hearing or speech impairments communicate effectively.  
- Bridges the gap between signers and non-signers.

---

## 🛠️ Tech Stack

**Frontend:** Streamlit (interactive web interface)  
**Backend:**
- OpenCV (video capture & processing)  
- MediaPipe (hand tracking & landmark detection)  
- NumPy (numerical operations)  
- scikit-learn RandomForestClassifier (gesture recognition)  
**Data Storage:** Pickle (saving/loading trained models)  
**Language:** Python  

---
## 📊 Workflow
1. **Data Collection** – Capture gesture data using `data_collection.py` → saves dataset as CSV/NumPy arrays.  
2. **Data Processing** – Clean and preprocess data with `data_processing.py` → generates normalized dataset.  
3. **Model Training** – Train and save the model using `model_training.py` → produces `model.pkl`.  
4. **Real-Time Detection** – Run `sign_detection.py` → detects gestures via webcam in real-time with live feedback.

---

## 🎯 Future Improvements

Expand dataset to include full ASL/ISL alphabets & words

Incorporate Deep Learning (CNN/RNN) for higher accuracy

Add gesture-to-speech output

Deploy as a web or mobile application for broader accessibility
---

## 🤝 Acknowledgments
- **MediaPipe** by Google for hand landmark detection  
- **OpenCV** for real-time computer vision  
- Inspired by real-world AI applications for accessibility

