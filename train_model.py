import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']

df = pd.read_csv('diabetes.csv', names=columns)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_accuracy = round(accuracy_score(y_test, lr_preds) * 100, 2)

print("Logistic Regression Accuracy:", lr_accuracy, "%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_preds))

# --- Model 2: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = round(accuracy_score(y_test, rf_preds) * 100, 2)

print("\nRandom Forest Accuracy:", rf_accuracy, "%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# --- Feature importance from Random Forest ---
importances = rf_model.feature_importances_
feature_names = X.columns.tolist()
feature_importance_dict = dict(zip(feature_names, [round(float(i)*100, 1) for i in importances]))

# Sort by importance
feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
print("\nFeature Importances:")
for k, v in feature_importance_dict.items():
    print(f"  {k}: {v}%")

# --- Save everything ---
with open('model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save accuracies and feature importance to a file
import json
model_info = {
    'lr_accuracy': lr_accuracy,
    'rf_accuracy': rf_accuracy,
    'feature_importance': feature_importance_dict
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f)

print("\nAll models and info saved!")