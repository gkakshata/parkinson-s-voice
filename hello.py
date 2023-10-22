import streamlit as st
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("parkinsons.csv")

# Attributes to take as input
input_attributes = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", 
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", 
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# Preprocess the dataset
X = df[input_attributes].values
Y = df['status'].values

# Standardize the features
ss = StandardScaler()
X = ss.fit_transform(X)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X, Y)

# Define the Streamlit app
st.title('Parkinson\'s Disease Prediction')
st.write("Please enter the values for the following attributes:")

# Get user inputs for each attribute
input_data = []
for i, feature in enumerate(input_attributes):
    # Set the minimum value to -100 for all attributes except "spread1"
    min_value = -100 if feature != "spread1" else None
    user_input = st.number_input(f'Enter value for {feature}:', min_value=min_value)
    input_data.append(user_input)

# Predict on user input and display result
if st.button('Predict'):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    if prediction[0] == 0:
        st.success("Result: Negative, No Parkinson's")
    else:
        st.error("Result: Positive, Parkinson's Found")