# Breast Cancer Detection using Deep Learning and Flask

This repository contains the implementation of my final year project on **AI-assisted breast cancer detection using mammography**. The system fine-tunes a DenseNet121 model on the CBIS-DDSM dataset, integrates **Grad-CAM explainability**, and deploys the results via a **Flask web application**.

---

##  Features
- Fine-tuned DenseNet121 model for binary classification (benign vs malignant).
- Training and evaluation scripts with metrics (accuracy, AUC, F1-score).
- Flask-based web interface:
  - Secure login
  - DICOM-to-PNG conversion
  - Image upload & preview
  - Grad-CAM overlay for explainability
  - Magnifier tool & download support

---

project-root/
Model Training Code/
  Breast Cancer Detection Model + Grad-CAM.ipynb
flask_app/
  templates/
  static/
  app.py
  requirements.txt
README.md

## To Run Flask App 
pip install -r requirements.txt
cd flask_app
python app.py
Open browser at http://127.0.0.1:5000
Login infomation
USERNAME = "admin"
PASSWORD = "@Sonnaeun1"

Results
DenseNet121 achieved:
Accuracy:75.91%
AUC: 0.8113
Balanced recall across benign and malignant cases
Integrated Grad-CAM visualisation increased interpretability and user trust.

Video Demostration: https://www.dropbox.com/scl/fi/7xwvdgsb9nwm0aso47yrd/FYP-Video-Demo.mp4?rlkey=b5ugcodjdtxq04pco1y67495r&st=wwo9h3t6&dl=0
