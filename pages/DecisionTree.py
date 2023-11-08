import streamlit as st
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO
import pydotplus
from IPython.display import Image
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Title for the Streamlit app
st.title("Decision Tree")

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

# Parameters for the Decision Tree Classifier
param_dist = {
    'max_depth': [None] + list(randint(1, 30).rvs(10)),
    'min_samples_split': list(randint(2, 11).rvs(10)),
    'min_samples_leaf': list(randint(1, 5).rvs(10)),
    'max_features': [None, 'sqrt', 'log2']
}

# Create a sidebar for setting hyperparameters
st.sidebar.header("Set Hyperparameters")
max_depth = st.sidebar.slider("Max Depth", 1, 30, 6)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 5, 1)
max_features = st.sidebar.selectbox("Max Features", [None, 'sqrt', 'log2'])

# Initialize the Decision Tree Classifier with selected hyperparameters
decision_tree = DecisionTreeClassifier(max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features,
                                       random_state=42)

st.subheader("Evaluation:")

# Train the model on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Display the classification report in a more visually appealing format
classification_rep = classification_report(y_test, y_pred, output_dict=True)
classification_df = pd.DataFrame(classification_rep).T

st.subheader("Classification Report:")
st.table(classification_df)

# Display the decision tree
st.subheader("Decision Tree")
dot_data = StringIO()
export_graphviz(decision_tree, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=X.columns, class_names=car_evaluation.target_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Convert the graph to a Base64 image
image = graph.create(format='png')
st.image(image)
