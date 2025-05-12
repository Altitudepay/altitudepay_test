# 💳 BIN-based Processor Success Predictor

This is an interactive Streamlit app that predicts the top 3 best payment processors for a given BIN using a trained machine learning model.

## 🚀 Features

- Predict best processors using BIN input
- Choose between manual input or CSV upload
- View success rates, charts, and fallback logic
- Download results and view auto-logged reports

## 🧠 Tech Stack

- Python + Streamlit
- XGBoost (binary classification)
- Altair for data visualization

## 📦 How to Run Locally

```bash
git clone this repo
cd processor-predictor-app
pip install -r requirements.txt
streamlit run app.py
