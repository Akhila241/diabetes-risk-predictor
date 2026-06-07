# Diabetes Risk Predictor

A full stack machine learning web application that predicts diabetes risk based on patient health metrics.

## Live Demo
[Click here to view the live app](#) ← we'll add this after deployment

## Screenshots
![App Screenshot](screenshot.png)

## Tech Stack
- **ML:** scikit-learn (Logistic Regression, Random Forest)
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render

## Features
- Predicts diabetes risk with 74.68% accuracy
- Shows probability score with visual risk meter
- Displays real feature importance from Random Forest model
- Compares Logistic Regression vs Random Forest accuracy
- Clean responsive UI

## Dataset
Pima Indians Diabetes Dataset — 768 patient records, 8 features

## ML Models
| Model | Accuracy |
|---|---|
| Logistic Regression | 74.68% |
| Random Forest | 72.08% |

## How to Run Locally
1. Clone the repo
git clone https://github.com/Akhila241/diabetes-risk-predictor.git
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python train_model.py
4. Run the app
python app.py

5. Open `http://127.0.0.1:5000`

## Project Structure

├── app.py              # Flask backend
├── train_model.py      # ML model training
├── model.pkl           # Trained Logistic Regression model
├── rf_model.pkl        # Trained Random Forest model
├── model_info.json     # Model accuracies and feature importance
├── diabetes.csv        # Dataset
└── templates/
└── index.html      # Frontend

## Author
Akhila — [GitHub](https://github.com/Akhila241)


