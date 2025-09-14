# Sign Language Detection System

A Python-based real-time gesture recognition system using **Computer Vision** and **Machine Learning** to aid accessibility. This project detects hand gestures and converts them into text in real-time. It can be extended to recognize alphabets, words, or custom gestures.

---

## 🚀 Features

1. **Real-Time Gesture Recognition**  
   Tracks hand gestures using **MediaPipe’s hand landmarks** in live video.

2. **Webcam-Based Input**  
   Uses a standard webcam for capturing input—no additional hardware required.  
   Continuously captures frames to identify hand positions and shapes.

3. **Random Forest Gesture Classification**  
   Extracts 21 key hand landmark points as feature vectors.  
   Uses **Random Forest Classifier** from scikit-learn to classify gestures.  
   No deep learning (CNN) is required, making it lightweight.

4. **Text-Only Output**  
   Outputs recognized signs as on-screen text (no audio/speech synthesis).  
   Helps users understand the translated sign without relying on sound.

5. **Streamlit-Based User Interface**  
   Provides an interactive web interface built using **Streamlit**.  
   Displays:
   - Live webcam feed  
   - Hand detection visualization  
   - Detected gesture label in real-time

6. **Live Feedback via OpenCV**  
   Draws hand landmarks and bounding boxes over the live video feed.  
   Useful during debugging and proper hand positioning.

7. **Trainable and Extendable**  
   Add more gestures by collecting data and retraining the model.  
   Supports dataset handling using **NumPy** and **Pandas**.

8. **Accessibility-Focused**  
   Aims to bridge communication gaps for individuals with hearing or speech impairments.  
   Converts gestures into text, aiding interaction with non-signers.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3.x  
- **Libraries:** OpenCV, MediaPipe, NumPy, Pandas, scikit-learn, Streamlit  
- **Model:** Random Forest Classifier for gesture recognition  

---

## 📊 Workflow
1. **Data Collection** – Capture gesture data using `data_collection.py` → saves dataset as CSV/NumPy arrays.  
2. **Data Processing** – Clean and preprocess data with `data_processing.py` → generates normalized dataset.  
3. **Model Training** – Train and save the model using `model_training.py` → produces `model.pkl`.  
4. **Real-Time Detection** – Run `sign_detection.py` → detects gestures via webcam in real-time with live feedback.

---

## 🎯 Results
- Accurate recognition of trained gestures in real-time  
- Demonstrates **Computer Vision + Machine Learning** for accessibility applications  
- Can be extended into a **Sign-to-Text** or **Sign-to-Speech** system  

---

## 📌 Future Improvements
- Expand dataset to include full **ASL/ISL alphabets & words**  
- Incorporate **Deep Learning (CNN/RNN)** for higher accuracy  
- Deploy via **Streamlit WebApp**  
- Add **gesture-to-speech output** for a complete accessibility solution  
---

## 🤝 Acknowledgments
- **MediaPipe** by Google for hand landmark detection  
- **OpenCV** for real-time computer vision  
- Inspired by real-world AI applications for accessibility

