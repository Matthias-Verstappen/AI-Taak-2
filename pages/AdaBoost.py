import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo

# Streamlit UI
st.title("AdaBoost")

# Load the data and perform the preprocessing steps
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
n_estimators = st.sidebar.slider("Number of Estimators", 1, 200, 100, step=1)
max_depth = st.sidebar.slider("Decision Tree Max Depth", 1, 10, 6, step=1)

# Initialize the AdaBoost Classifier with the selected hyperparameters
ada_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators, random_state=42)

# Train the model on the training data
ada_boost.fit(X_train, y_train)

# Make predictions on the test data
y_pred_ada = ada_boost.predict(X_test)

# Calculate accuracy
accuracy_ada = accuracy_score(y_test, y_pred_ada)

# Display results in Streamlit
st.subheader("Evaluation:")
st.write(f"AdaBoost Accuracy: {accuracy_ada:.2f}")

st.subheader("Classification Report:")
classification_rep = classification_report(y_test, y_pred_ada, output_dict=True)
classification_df = pd.DataFrame(classification_rep).T
st.dataframe(classification_df)
