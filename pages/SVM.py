import streamlit as st
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Streamlit UI
st.title("Support Vector Machine (SVM)")

# Fetch the car evaluation dataset (ID 19)
car_evaluation = fetch_ucirepo(id=19)

# Extract the features (X) and targets (y) as pandas dataframes
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Define categorical columns for Label Encoding
categorical_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# Label Encode the categorical features
label_encoders = {}  # Dictionary to store label encoders for each categorical feature

for column in categorical_columns:
    label_encoder = LabelEncoder()
    X.loc[:, column] = label_encoder.fit_transform(X[column])  # Encode the feature
    label_encoders[column] = label_encoder  # Store the label encoder for later use

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a sidebar for setting hyperparameters
st.sidebar.header("Set Hyperparameters")
best_C = st.sidebar.slider("C (Regularization Parameter)", 0.1, 10.0, 1.0, step=0.1)
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

# Initialize the Support Vector Machine Classifier with the selected hyperparameters
svm_classifier = SVC(C=best_C, kernel=kernel, random_state=42)

# Train the model on the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Display results in Streamlit
st.subheader("Evaluation:")
st.write(f"SVM Accuracy: {accuracy_svm:.2f}")

st.subheader("Classification Report:")
classification_rep = classification_report(y_test, y_pred_svm, output_dict=True)
classification_df = pd.DataFrame(classification_rep).T
st.dataframe(classification_df)
