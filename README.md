# Smart-Traffic-System-YOLO-BEV_ASEP2
Real-time traffic density &amp; smart signal system using YOLO and BEV

IMPQ : These are the instructions for further my team members go through the readme i have made and you will get what what to do 

# 🚦 Smart Traffic Monitoring System (YOLO + SORT + BEV + Heatmap)   (till now this much progress is done by pk2807)

A computer vision-based traffic analysis system that detects, tracks, and counts vehicles using YOLOv8 and SORT.  
It also provides traffic density estimation and heatmap visualization.

---

## 🎥 Demo Video

👉 https://drive.google.com/drive/folders/1tQHxOUryJvcnEer81LOL6YC8mtuKzNF8

---

## 📁 Dataset / Videos

⚠️ Videos are not included in this repository due to large file size.

Download videos from:
👉 https://drive.google.com/drive/folders/1tQHxOUryJvcnEer81LOL6YC8mtuKzNF8

After downloading, place them inside the `data/` folder and rename exactly as:

traffic_main.mp4  
traffic_test.mp4  

---

## 🚀 Features

- Vehicle Detection using YOLOv8  
- Multi-object Tracking using SORT  
- Unique Vehicle Counting (Line Crossing)  
- Region of Interest (ROI) filtering  
- Bird’s Eye View (BEV) transformation  
- Traffic Density Detection (LOW / MEDIUM / HIGH)  
- Heatmap Visualization  

---

## 🛠️ Installation

pip install -r requirements.txt

---

## ▶️ Run the Project

python detection/main1.py

---

## 📂 Project Structure

Smart-Traffic-System/
│
├── detection/
│   ├── main1.py
│   ├── sort.py
│
├── data/
│
├── requirements.txt
├── README.md
├── .gitignore

---

## ⚠️ Important Notes

- Do NOT upload:
  - .venv/
  - .pt model files
  - .mp4 videos

- These are handled using .gitignore

---

## 👥 Instructions for Team Members

1. Clone repository:
git clone <your-repo-link>

2. Install dependencies:
pip install -r requirements.txt

3. Download videos from Google Drive

4. Place videos inside:
data/

5. Rename correctly:
traffic_main.mp4  
traffic_test.mp4  

6. Run project:
python detection/main1.py

---

## 📊 Output

- Vehicle count  
- Tracking IDs  
- Heatmap visualization  
- BEV (top view)  
- Traffic density status  

---

## 🧠 Future Improvements

- Improve tracking using DeepSORT  
- Vehicle speed estimation  
- Lane detection  
- Traffic violation detection  

---

## 📌 Summary

Detection → Tracking → Counting → BEV → Heatmap → Density Analysis

---

🔥 Clean, simple, and ready for further development.
