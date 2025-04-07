from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# Load and preprocess dataset
df = pd.read_csv("lettuce_dataset_updated.csv", encoding='ISO-8859-1')
X = df[["Temperature (°C)", "Humidity", "TDS Value (ppm)", "pH Level"]]
y = df["Growth Days"]  # Adjust if needed

# Encode target labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X, y_encoded)

# Save the model and encoder
joblib.dump(clf, "lettuce_growth_classifier.pkl")
joblib.dump(label_encoder, "growth_stage_label_encoder.pkl")
