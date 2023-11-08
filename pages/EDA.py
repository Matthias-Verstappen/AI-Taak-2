import streamlit as st
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# Streamlit UI
st.title("Exploratory Data Analysis (EDA)")

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

# Display the first 5 rows of the features after Label Encoding
st.subheader("First 5 rows of the features after Label Encoding:")
st.dataframe(X.head())

# Display unique values for each feature
st.subheader("Unique values for each feature:")
for column in X.columns:
    unique_values = X[column].unique()
    decoded_values = label_encoders[column].inverse_transform(unique_values.astype(int))  # Decode the unique values
    st.write(f"**Feature '{column}':**")
    st.write(f"Unique Values: {unique_values}")
    st.write(f"Decoded Values: {decoded_values}")

# Calculate summary statistics for numerical features
st.subheader("Summary statistics for numerical features:")
st.dataframe(X.describe())

# Count missing values in each column
missing_values = X.isnull().sum()
st.subheader("Missing values in each column:")
st.dataframe(missing_values)

# Check for imbalance in categorical features
for column in categorical_columns:
    st.subheader(f"Value counts for '{column}' column:")
    st.dataframe(X[column].value_counts())
