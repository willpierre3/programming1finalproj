import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load in the dataset
s = pd.read_csv("social_media_usage.csv")

# Define a function to clean the data
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Clean the data to remove excess information
ss = s
ss["sm_li"] = clean_sm(ss["web1h"])
ss["income"] = np.where(ss["income"] <= 9, ss["income"], np.nan)
ss["education"] = np.where(ss["educ2"] <= 8, ss["educ2"], np.nan)
ss["parent"] = clean_sm(ss["par"])
ss["married"] = clean_sm(ss["marital"])
ss["female"] = clean_sm(ss["gender"])
ss["age"] = np.where(ss["age"] <= 98, ss["age"], np.nan)
ss = ss.dropna()

# Select the features to be used in the model
X = ss[["income", "education", "parent", "married", "female", "age"]]
y = ss["sm_li"]

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, stratify=y, test_size=0.2, random_state=999
)

# Train a logistic regression model using the data
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)

# Create name for Streamlit
st.title("LinkedIn User Prediction App")

# Enter values for the variables to make predictions
st.header("Enter User Information")
income = st.slider("Income Level (1-9)", min_value=1, max_value=9, value=5)
education = st.slider("Education Level (1-8)", min_value=1, max_value=8, value=4)
parent = st.selectbox("Parent?", ["No", "Yes"])
married = st.selectbox("Married?", ["No", "Yes"])
female = st.selectbox("Female?", ["No", "Yes"])
age = st.slider("Age", min_value=18, max_value=98, value=30)

# Convert the inputs to model format
parent = 1 if parent == "Yes" else 0
married = 1 if married == "Yes" else 0
female = 1 if female == "Yes" else 0

person = [income, education, parent, married, female, age]

# Use the logistic regression model to make a prediction
predicted_class = lr.predict([person])[0]
probs = lr.predict_proba([person])

# Return the prediction decision
st.subheader("Prediction")
if predicted_class == 1:
    st.success("This person is predicted to be a LinkedIn user.")
else:
    st.error("This person is predicted to NOT be a LinkedIn user.")

# Show the probabilities of the outcome
st.subheader("Prediction Probabilities")
st.write(f"Probability of being a LinkedIn user: {probs[0][1]:.2f}")
st.write(f"Probability of NOT being a LinkedIn user: {probs[0][0]:.2f}")
