#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle

# Load the pickled Logistic Regression model
with open('logisticmodel.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_chd_probability(age, prevalent_hyp, diabetes, totchol, sysbp):
    # Prepare the input data as a 2D array for prediction
    input_data = [[age, prevalent_hyp, diabetes, totchol, sysbp]]
    # Predict the probability of tenyearCHD using the loaded model
    prediction = model.predict_proba(input_data)[:, 1]
    return prediction[0]

def main():
    st.title('CHD Probability Prediction')
    st.write('Enter the values of the independent features to predict the probability of tenyearCHD.')

    # Get user inputs for the independent features
    age = st.slider('Age', 20, 100, 50)
    prevalent_hyp = st.selectbox('Prevalent Hypertension (0 for No, 1 for Yes)', [0, 1])
    diabetes = st.selectbox('Diabetes (0 for No, 1 for Yes)', [0, 1])
    totchol = st.slider('Total Cholesterol', 100, 500, 200)
    sysbp = st.slider('Systolic Blood Pressure', 80, 250, 120)

    if st.button('Predict Probability'):
        # Calculate the probability of tenyearCHD using the model
        probability = predict_chd_probability(age, prevalent_hyp, diabetes, totchol, sysbp)
        st.write(f'The probability of tenyearCHD is: {probability:.2f}')

if __name__ == '__main__':
    main()


# In[ ]:




