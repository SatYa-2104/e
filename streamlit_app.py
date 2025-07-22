import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import openai

# Optional: Use Streamlit secrets for secure API key storage
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

# Load trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found! Please upload 'model.pkl'.")

st.markdown("<h1 style='color:green;'>💼 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("Enter employee details below to predict their expected salary.")

# Input form
with st.form("salary_form"):
    st.markdown("### 📝 Employee Information")

    experience = st.slider("Years of Experience", 0, 30, 2)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    role = st.selectbox("Role", ["Intern", "Junior Engineer", "Senior Engineer", "Manager", "Director"])
    location = st.selectbox("Location", ["India", "USA", "UK", "Germany", "Other"])

    submit = st.form_submit_button("Predict Salary")

def encode_input(experience, education, role, location):
    education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    role_map = {"Intern": 0, "Junior Engineer": 1, "Senior Engineer": 2, "Manager": 3, "Director": 4}
    location_map = {"India": 0, "USA": 1, "UK": 2, "Germany": 3, "Other": 4}
    return [experience, education_map[education], role_map[role], location_map[location]]

# Predict and display output
if submit and os.path.exists(model_path):
    input_data = np.array(encode_input(experience, education, role, location)).reshape(1, -1)
    predicted_salary = model.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: ₹{predicted_salary:,.2f}")

    st.markdown("### 📊 Salary Breakdown")
    base_salary = 30000
    exp_bonus = experience * 2000
    edu_bonus = [0, 5000, 10000, 15000][encode_input(experience, education, role, location)[1]]
    role_bonus = [0, 5000, 10000, 15000, 20000][encode_input(experience, education, role, location)[2]]
    loc_bonus = [0, 10000, 8000, 12000, 5000][encode_input(experience, education, role, location)[3]]
    values = [base_salary, exp_bonus, edu_bonus, role_bonus, loc_bonus]
    labels = ["Base", "Experience", "Education", "Role", "Location"]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color="green")
    ax.set_ylabel("Amount (₹)")
    st.pyplot(fig)

# AI Chat Assistant using OpenAI
st.markdown("---")
st.markdown("<h2 style='color:green;'>🤖 AI Chat Assistant</h2>", unsafe_allow_html=True)
user_query = st.text_input("Ask me anything about career, salaries, roles, etc:")
if user_query and openai.api_key:
    try:
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_query}]
            )
        reply = response['choices'][0]['message']['content']
        st.markdown("**AI Response:**")
        st.write(reply)
    except Exception as e:
        st.error(f"API Error: {e}")
elif user_query:
    st.warning("OpenAI API key not found. Set `OPENAI_API_KEY` in secrets or env variables.")