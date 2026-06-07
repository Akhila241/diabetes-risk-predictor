import pandas as pd

# Column names for this dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']

# Load the dataset
df = pd.read_csv('diabetes.csv', names=columns)

# Let's explore it
print("Shape of data:", df.shape)        # rows and columns
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic stats:")
print(df.describe())
print("\nHow many diabetic vs non-diabetic:")
print(df['Outcome'].value_counts())