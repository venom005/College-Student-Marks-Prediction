import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.write("""
# College Student Marks Prediction App
This app predicts the **Marks** scored by student
""")

st.sidebar.markdown('''
            By Pankaj Raina \n
            GitHub Repository for the [App](https://github.com/venom005/College-Student-Marks-Prediction)
    ''')
st.write('''
        **Assuming the study hours to be maximum 10 hours.**
        ''')
    

def user_input():
    hr = st.number_input("Number of Hours of actual Study",0.00,10.00,6.2)
    hr = np.array([[hr]]).astype(np.float64)
    return hr

pred = user_input()

#Loading dataset
df = pd.read_csv(r"C:\Users\RAINA\Downloads\Data.txt")


attr = df.iloc[:,:-1].values
labels = df.iloc[:,1].values

LR = LinearRegression()
LR.fit(attr,labels)

prediction = round(float(LR.predict(pred)),2)

if pred > 9.97:
    st.button("Predict")
    st.success("The Predicted Percentage is 100.00")
else:
    st.button("Predict")
    st.success("The Predicted Percentage is {}".format(prediction))
st.dataframe(df)
st.area_chart(df)

