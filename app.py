import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import requests
import os


from dotenv import load_dotenv
load_dotenv()
import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")


def query_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "moonshotai/kimi-k2-instruct-0905",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


@st.cache_resource
def load_model():
    if not os.path.exists("segment_model.pkl"):
        df = pd.read_csv("Mall_Customers.csv") 
        X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X)
        with open("segment_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
    return pickle.load(open("segment_model.pkl", "rb"))


model = load_model()


st.set_page_config(page_title="Mall Chatbot", page_icon="🛍️")
st.title("🛍️ Mall Customer Chatbot")
st.markdown("Get personalized mall service recommendations based on your profile.")

with st.form("input_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    income = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=60)
    score = st.slider("Spending Score (1-100)", 1, 100, 50)
    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    segment = model.predict([[age, income, score]])[0]
    prompt = f"A mall customer belongs to segment {segment} based on age {age}, income {income}k$, and spending score {score}. Suggest personalized services or offers."
    with st.spinner("Thinking with Groq..."):
        reply = query_groq(prompt)
    st.success("Here's your personalized recommendation:")
    st.write(reply)
